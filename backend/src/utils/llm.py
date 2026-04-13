import os
import gc
from threading import Lock, Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import torch
import warnings
from src.utils.app_config import AppConfig
from src.utils.logger import log_user_input, log_agent_response

warnings.filterwarnings('ignore')

class LLMManager:
    def __init__(self):
        self.llm_config = AppConfig.get_llm_config()
        self.model_list = self.llm_config.get("model_list", [])
        self.default_model_id = self.llm_config.get("default_model", "qwen-3-8b")
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.current_model_id = None
        self.enable_thinking = False
        self.system_prompt = AppConfig.get_agents_config().get("financial_assistant", {}).get("instructions", "Bạn là chuyên gia tài chính AI.")
        self.bad_words_ids = None
        self._load_lock = Lock()

        # Do not eagerly load a model here.
        # Eager loading blocks lightweight endpoints (e.g. /api/chat/models) on first request.
        self.current_model_id = self.default_model_id

    def ensure_loaded(self) -> bool:
        """Ensure a model is loaded; returns True when ready."""
        if self.model is not None and self.tokenizer is not None:
            return True
        return self.switch_model(self.current_model_id or self.default_model_id)

    def _cleanup_vram(self):
        """Unload current model and free VRAM."""
        if self.model is not None:
            print(f"Unloading model: {self.current_model_id}...")
            del self.model
            del self.tokenizer
            if self.processor is not None:
                del self.processor
            self.model = None
            self.tokenizer = None
            self.processor = None
            
            # Aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("VRAM cleared.")

    def get_available_models(self):
        return [{"id": m["id"], "name": m["name"]} for m in self.model_list]

    def switch_model(self, model_id: str):
        with self._load_lock:
            if self.current_model_id == model_id and self.model is not None and self.tokenizer is not None:
                return True

            # Find model config
            model_cfg = next((m for m in self.model_list if m["id"] == model_id), None)
            if not model_cfg:
                print(f"Model ID {model_id} not found in config.")
                return False

            self._cleanup_vram()

            model_name = model_cfg["huggingface_model"]
            self.enable_thinking = model_cfg.get("enable_thinking", False)
            self.current_model_id = model_id
            model_kind = (model_cfg.get("model_kind") or "causal_lm").lower()
            quant = (model_cfg.get("quantization") or "4bit").lower()
            
            print(f"Loading model: {model_name}...")
            
            bnb_config = None
            if quant == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                # 4-bit Quantization (default)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

            try:
                if model_kind == "multimodal":
                    # Try CausalLM first (text-only). This often loads faster and avoids
                    # optional vision/video deps (e.g., torchvision) when you don't need multimodal.
                    try:
                        self.processor = None
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            quantization_config=bnb_config,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True
                        )
                    except Exception:
                        # Fallback to multimodal model
                        try:
                            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                            self.tokenizer = self.processor.tokenizer
                        except Exception as e:
                            print(f"Processor load failed (will fall back to tokenizer-only): {e}")
                            self.processor = None
                            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                        self.model = AutoModelForMultimodalLM.from_pretrained(
                            model_name,
                            quantization_config=bnb_config,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True
                        )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    )
                
                if model_cfg.get("bad_words_filter", True):
                    self.bad_words_ids = self._get_non_vietnamese_bad_words()
                else:
                    self.bad_words_ids = None
                print(f"Model {model_id} loaded successfully!")
                return True
            except Exception as e:
                print(f"Failed to load model {model_id}: {str(e)}")
                return False

    def _get_non_vietnamese_bad_words(self):
        bad_words = []
        if not self.tokenizer or not hasattr(self.tokenizer, 'vocab_size'):
            return None

        def is_allowed_char(ch):
            if ord(ch) < 128: return True
            if '\u00c0' <= ch <= '\u01b0' or '\u1ea0' <= ch <= '\u1ef9': return True
            return False

        try:
            for i in range(self.tokenizer.vocab_size):
                token = self.tokenizer.decode([i])
                if any(not is_allowed_char(ch) for ch in token):
                    bad_words.append([i])
            return bad_words
        except Exception:
            return None

    def generate_response(self, user_input, history=[]):
        if not self.ensure_loaded():
            yield "[Error] Failed to load model."
            return

        # Format history
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            messages.append(msg)
        
        log_user_input(user_input)
        messages.append({"role": "user", "content": user_input})

        # Get specific model config for generation params
        model_cfg = next((m for m in self.model_list if m["id"] == self.current_model_id), {})
        model_kind = (model_cfg.get("model_kind") or "causal_lm").lower()
        
        if model_kind == "multimodal":
            if self.processor is not None:
                try:
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                except TypeError:
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                model_inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
                streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            else:
                # Tokenizer-only fallback (text chat)
                try:
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    # Very last resort
                    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        else:
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if "qwen-3" in (self.current_model_id or "").lower():
                template_kwargs["enable_thinking"] = self.enable_thinking

            try:
                text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=model_cfg.get("max_new_tokens", 2048),
            do_sample=True,
            temperature=model_cfg.get("temperature", 0.7),
            top_p=model_cfg.get("top_p", 0.8),
            top_k=model_cfg.get("top_k", 20),
            repetition_penalty=1.1,
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            eos_token_id=self.tokenizer.eos_token_id
        )

        if self.bad_words_ids:
            generation_kwargs['bad_words_ids'] = self.bad_words_ids

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = ""
        strip_thought = bool(model_cfg.get("strip_thought", False))
        stop_markers = (
            "Thinking Process",
            "Review Constraints",
            "Analyze the Request",
            "Final Output Generation",
            "chain-of-thought",
            "Thought:",
            "thought",
        )

        if not strip_thought:
            for new_text in streamer:
                full_response += new_text
                yield new_text
        else:
            pending = ""
            tail_keep = 80

            def find_marker(s: str) -> int:
                hits = [s.find(m) for m in stop_markers]
                hits = [h for h in hits if h != -1]
                return min(hits) if hits else -1

            for new_text in streamer:
                pending += new_text
                idx = find_marker(pending)
                if idx != -1:
                    safe = pending[:idx].rstrip()
                    if safe:
                        full_response += safe
                        yield safe
                    pending = ""
                    break

                if len(pending) > tail_keep:
                    emit = pending[:-tail_keep]
                    pending = pending[-tail_keep:]
                    if emit:
                        full_response += emit
                        yield emit

            if pending.strip():
                full_response += pending
                yield pending
        
        log_agent_response(f"Assistant ({self.current_model_id})", full_response)

# Singleton instance
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMManager()
    return _llm_instance
