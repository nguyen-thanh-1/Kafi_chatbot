# LLM Models

Thư mục chứa các LLM model wrappers.

## Files

- **Llama_3_1_8B_Instruct.py** - Llama 3.1 8B wrapper (version 1)
- **Llama_3_1_8B_Instruct_v2.py** - Llama 3.1 8B wrapper (version 2, lazy loading)
- **Qwen2.5_14B_Instruct.py** - Qwen 2.5 14B wrapper
- **Qwen2.5_7B_Instruct.py** - Qwen 2.5 7B wrapper

## Usage

```python
# Llama 3.1 8B (lazy loading)
from llm_models.Llama_3_1_8B_Instruct_v2 import generate_response

response = generate_response(
    user_input="What is NLP?",
    system_prompt="You are a helpful assistant.",
    max_new_tokens=512,
    temperature=0.2
)
```

## Notes

- **Llama_3_1_8B_Instruct_v2.py** uses lazy loading (recommended)
- All models support 4-bit quantization
- Models auto-detect GPU availability
