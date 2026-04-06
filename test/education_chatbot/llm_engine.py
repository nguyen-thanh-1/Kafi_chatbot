"""
LLM Engine - Wrapper cho Llama 3.1 8B
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
from typing import List, Generator


class EducationalLLM:
    """Llama 3.1 8B với streaming support"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        if self._initialized:
            return
        
        print(f"Loading LLM: {model_id}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        self._initialized = True
        print("LLM loaded successfully!")
    
    def generate(
        self,
        user_input: str,
        system_prompt: str = None,
        history: List[dict] = None,
        max_tokens: int = 700,
        temperature: float = 0.21
    ) -> str:
        """Generate response (non-streaming)"""
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        
        if history is None:
            history = []
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                         skip_special_tokens=True)
        
        return response.strip()
    
    def stream_generate(
        self,
        user_input: str,
        system_prompt: str = None,
        history: List[dict] = None,
        max_tokens: int = 700,
        temperature: float = 0.21
    ) -> Generator[str, None, None]:
        """Generate response with streaming"""
        
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."
        
        if history is None:
            history = []
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
