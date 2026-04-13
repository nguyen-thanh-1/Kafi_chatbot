"""
Gemma 4 E4B - Interactive Multimodal Runner (OPTIMIZED)
======================================================
Chạy mô hình Gemma 4 E4B (instruction-tuned) trên GPU 16GB VRAM.
Đã tối ưu: 8-bit Quantization + Streaming + Token/s Counter.
"""

import argparse
import sys
import time
import torch
import threading
import warnings
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForMultimodalLM, 
    BitsAndBytesConfig,
    TextIteratorStreamer
)

# Ẩn các cảnh báo kỹ thuật không cần thiết
warnings.filterwarnings("ignore", message="MatMul8bitLt")
warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID_BASE = "google/gemma-4-E4B"
MODEL_ID_IT = "google/gemma-4-E4B-it"

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là một trợ lý AI thông minh, hữu ích và thân thiện. "
    "Hãy trả lời bằng tiếng Việt khi người dùng hỏi bằng tiếng Việt."
)

GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "do_sample": True,
}


def load_model(use_base=False, verbose=True):
    """Tải Gemma 4 E4B model với 8-bit quantization."""
    model_id = MODEL_ID_BASE if use_base else MODEL_ID_IT

    if verbose:
        print(f"\n{'='*60}")
        print(f"  🚀 LOADING: {model_id}")
        print(f"  Precision: 8-bit Int8 (Quality Mode)")
        print(f"{'='*60}\n")

    start_time = time.time()

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✅ Model loaded in {elapsed:.1f}s")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   VRAM Usage: {allocated:.1f} GB / {total:.1f} GB")
    return model, processor


def chat_streaming(model, processor, messages, enable_thinking=False):
    """Sinh câu trả lời ở chế độ Streaming và tính Token/s."""
    
    # 1. Chuẩn bị Input
    has_multimodal = any(
        isinstance(m.get('content'), list) or 
        (isinstance(m.get('content'), dict) and m['content'].get('type') in ['image', 'audio'])
        for m in messages
    )

    if has_multimodal:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(model.device)
    else:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)

    # 2. Setup Streamer
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 3. Chạy generation
    generation_kwargs = dict(inputs, streamer=streamer, **GENERATION_CONFIG)
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 4. In kết quả và tính toán
    print("\n🤖 Gemma 4: ", end="", flush=True)
    full_response = ""
    token_count = 0
    
    start_time = None
    
    for new_text in streamer:
        if start_time is None:
            start_time = time.time() # Bắt đầu tính giờ từ token đầu tiên
        
        print(new_text, end="", flush=True)
        full_response += new_text
        token_count += 1 # Ước tính token thô dựa vào số lần yield của streamer
    
    elapsed = time.time() - start_time if start_time else 0
    
    # Đếm token chính xác bằng tokenizer
    actual_tokens = len(processor.tokenizer.encode(full_response, add_special_tokens=False))
    tps = actual_tokens / elapsed if elapsed > 0 else 0
    
    print(f"\n\n  ⏱️  {elapsed:.1f}s | {actual_tokens} tokens | {tps:.1f} tok/s")
    
    return full_response


def run_image_analysis(model, processor, image_source, question=None, enable_thinking=False):
    if question is None: question = "Mô tả chi tiết hình ảnh này."
    content = []
    if image_source.startswith("http"):
        content.append({"type": "image", "url": image_source})
    else:
        content.append({"type": "image", "image": Image.open(image_source)})
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    return chat_streaming(model, processor, messages, enable_thinking)


def interactive_chat(model, processor, enable_thinking=False, system_prompt=None):
    if system_prompt is None: system_prompt = DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n{'═'*60}")
    print(f"  🌟 Gemma 4 E4B Interactive Chat (8-bit Mode)")
    print(f"  Thinking: {'ON' if enable_thinking else 'OFF'} | Tok/s: Enabled")
    print(f"{'═'*60}\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input: continue
        if user_input.lower() in ("/quit", "/exit"): break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("🗑️ History cleared.\n")
            continue

        if user_input.lower().startswith("/image "):
            path = user_input[7:].strip()
            run_image_analysis(model, processor, path, enable_thinking=enable_thinking)
            print()
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            response_text = chat_streaming(model, processor, messages, enable_thinking)
            messages.append({"role": "assistant", "content": response_text})
            print()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            messages.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ Cần GPU NVIDIA."); return

    model, processor = load_model(use_base=args.base)
    interactive_chat(model, processor, enable_thinking=args.thinking)

if __name__ == "__main__":
    main()
