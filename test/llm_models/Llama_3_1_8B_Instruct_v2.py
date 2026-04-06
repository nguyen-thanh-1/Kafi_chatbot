import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Global variables for lazy loading
_model = None
_tokenizer = None

def _load_model():
    """Lazy load model - only load when first called"""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    print("Loading model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    print("Model loaded!")
    
    return _model, _tokenizer

def generate_response(user_input, history=None, system_prompt=None, max_new_tokens=512, temperature=0.2):
    # Lazy load model on first call
    model, tokenizer = _load_model()
    
    if history is None:
        history = []
        if system_prompt:
             history.append({"role": "system", "content": system_prompt})
    
    # Add user input to history if provided (for the chat template)
    # Note: The caller might manage history outside. 
    # If this function is stateless regarding history, we just use what's passed.
    messages = list(history) # Copy to avoid modifying original list in place immediately if not desired, or append specific for this run
    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    for token in streamer:
        # print(token, end="", flush=True) # Optional: callback or yield if needed
        response += token
    
    return response

if __name__ == "__main__":
    system_prompt = """You are an educational AI tutor.
Explain clearly.
Answer in Vietnamese.
Be accurate and step-by-step.
"""
    
    history = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        print("Bot: ", end="", flush=True)
        
        # Manually invoke logic so we can stream print here (duplicating logic slightly or using the function but without streaming)
        # OR just call generate_response and print result at end (less interactive but simple)
        # OR better: Refactor generate_response to yield, but for now let's just make it work.
        
        response = generate_response(user_input, history, max_new_tokens=512, temperature=0.2)
        print(response)
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
