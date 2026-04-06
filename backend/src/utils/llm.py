import os
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch
import warnings
from src.utils.app_config import AppConfig
from src.utils.logger import log_user_input, log_agent_response

warnings.filterwarnings('ignore')

class QwenLLM:
    def __init__(self):
        llm_config = AppConfig.get_llm_config()
        model_name = llm_config.get("model_list", [{}])[0].get("huggingface_model", "Qwen/Qwen3-8B")
        self.enable_thinking = llm_config.get("model_list", [{}])[0].get("enable_thinking", False)
        
        print(f"Loading model: {model_name}...")
        
        # 4-bit Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        agents_config = AppConfig.get_agents_config()
        self.system_prompt = agents_config.get("financial_assistant", {}).get("instructions", "Bạn là chuyên gia tài chính AI.")
        
        self.bad_words_ids = self._get_non_vietnamese_bad_words()
        print("Model loaded successfully!")

    def _get_non_vietnamese_bad_words(self):
        bad_words = []
        if not hasattr(self.tokenizer, 'vocab_size'):
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
        # Format history for Qwen3
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            messages.append(msg)
        
        # Log the user message nicely in the console
        log_user_input(user_input)

        messages.append({"role": "user", "content": user_input})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        if self.bad_words_ids:
            generation_kwargs['bad_words_ids'] = self.bad_words_ids

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = ""
        for new_text in streamer:
            full_response += new_text
            yield new_text
        
        # Log the final response in a panel
        log_agent_response("Financial Assistant", full_response)

# Singleton instance
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = QwenLLM()
    return _llm_instance
