import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread

class EducationalLLM:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading model: {model_id}...")
        
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
        print("Model loaded successfully!")

    def stream_chat(self, user_input, system_prompt, history=[]):
        """
        Streams the chat response.
        Args:
            user_input: The user's current valid input.
            system_prompt: The context-aware system instructions.
            history: List of previous conversation turns (dicts).
        Yields:
            str: Generated tokens.
        """
        # Construct messages
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]
        
        # Apply strict templating
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
            max_new_tokens=2048,
            temperature=0.2, # Low temp for educational precision
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
