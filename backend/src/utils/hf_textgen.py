from __future__ import annotations

import gc
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class TextGenResult:
    text: str
    raw: str


class HFTextGen:
    def __init__(
        self,
        model_name: str,
        *,
        quantization: str = "4bit",
        torch_dtype: Any = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.quantization = (quantization or "none").lower()
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        self._lock = Lock()
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def unload(self) -> None:
        with self._lock:
            if self._model is None and self._tokenizer is None:
                return

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True

        with self._lock:
            if self.is_loaded:
                return True

            bnb_config = None
            if torch.cuda.is_available():
                if self.quantization == "8bit":
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                elif self.quantization == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
                torch_dtype=self.torch_dtype,
                quantization_config=bnb_config,
                trust_remote_code=self.trust_remote_code,
            )
            return True

    def generate_chat(
        self,
        *,
        system: str,
        user: str,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
        stop_at_newline: bool = True,
    ) -> TextGenResult:
        if not self.ensure_loaded():
            return TextGenResult(text="", raw="")

        assert self._model is not None
        assert self._tokenizer is not None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"System: {system}\nUser: {user}\nAssistant:"

        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)

        # Deterministic classification-style decoding
        # Deterministic decoding: avoid passing sampling-only flags (e.g., temperature) to prevent warnings.
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            pad_token_id=(self._tokenizer.pad_token_id or self._tokenizer.eos_token_id),
            eos_token_id=self._tokenizer.eos_token_id,
        )

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True)
        text = raw.strip()
        if stop_at_newline and "\n" in text:
            text = text.split("\n", 1)[0].strip()
        return TextGenResult(text=text, raw=raw)
