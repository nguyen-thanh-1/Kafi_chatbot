# Fix: LLM vẫn chạy CPU - Lazy Loading

## 🔴 Vấn đề tiếp theo

Mặc dù đã đổi thứ tự load trong `rag_query.py`, LLM vẫn chạy CPU.

## 🔍 Nguyên nhân thực sự

**File `Llama_3_1_8B_Instruct_v2.py` load model NGAY KHI IMPORT:**

```python
# File: Llama_3_1_8B_Instruct_v2.py
import torch
from transformers import ...

# ❌ Load ngay khi import (dòng 12-28)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(...)  # Load ngay!
print("Model loaded!")

def generate_response(...):
    # Dùng model đã load
```

**Vấn đề:**
1. Khi `rag_query.py` gọi `get_llm()` → Import `Llama_3_1_8B_Instruct_v2`
2. Import → **Model load NGAY** (không đợi gọi function)
3. Lúc này nếu có models khác đã chiếm VRAM → LLM bị đẩy xuống CPU

**Thứ tự thực tế:**
```python
# rag_query.py
get_llm()  # Import Llama_3_1_8B_Instruct_v2
           # → Model load NGAY tại đây!
           # → Nếu VRAM đã bị chiếm → CPU

get_embedder()  # Load sau
get_reranker()  # Load sau
```

---

## ✅ Giải pháp: Lazy Loading

**Chỉ load model khi GỌI function lần đầu, không phải khi import:**

### File: `Llama_3_1_8B_Instruct_v2.py`

**Trước (Eager Loading):**
```python
# ❌ Load ngay khi import
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(...)
print("Model loaded!")

def generate_response(...):
    # Dùng model đã load
    inputs = tokenizer(prompt).to(model.device)
```

**Sau (Lazy Loading):**
```python
# ✅ Chỉ khai báo biến global
_model = None
_tokenizer = None

def _load_model():
    """Lazy load - chỉ load khi gọi lần đầu"""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer  # Đã load rồi
    
    # Load model (chỉ chạy 1 lần)
    print("Loading model...")
    _model = AutoModelForCausalLM.from_pretrained(...)
    _tokenizer = AutoTokenizer.from_pretrained(...)
    print("Model loaded!")
    
    return _model, _tokenizer

def generate_response(...):
    # Load model khi gọi function lần đầu
    model, tokenizer = _load_model()
    
    # Dùng model
    inputs = tokenizer(prompt).to(model.device)
```

---

## 📊 So sánh

### Eager Loading (Trước):
```python
# rag_query.py
from rag_pro_v2 import get_llm

get_llm()  # Import Llama_3_1_8B_Instruct_v2
           # → "Loading model..." (load NGAY)
           # → Nếu VRAM bị chiếm → CPU ❌
```

### Lazy Loading (Sau):
```python
# rag_query.py
from rag_pro_v2 import get_llm

get_llm()  # Import Llama_3_1_8B_Instruct_v2
           # → Không load gì cả (chỉ import)
           
# Khi query lần đầu:
rag.query("NLP là gì?")
  → generate_response()
    → _load_model()  # Load BÂY GIỜ
    → "Loading model..." (load lúc này)
    → Chiếm GPU ✅
```

---

## 🎯 Lợi ích Lazy Loading

### 1. **Kiểm soát thời điểm load**
```python
# Có thể load ĐÚNG LÚC cần
get_llm()        # Chỉ import, chưa load
get_embedder()   # Load embedding
get_reranker()   # Load reranker

# Query lần đầu → LLM mới load
rag.query(...)   # Load LLM BÂY GIỜ → GPU
```

### 2. **Tránh load không cần thiết**
```python
# Nếu chỉ import nhưng không dùng
from Llama_3_1_8B_Instruct_v2 import generate_response
# → Không load gì (tiết kiệm thời gian)

# Chỉ load khi thực sự gọi
generate_response("hello")  # Load lúc này
```

### 3. **Dễ debug**
```python
# Biết chính xác khi nào model load
print("Before import")
from Llama_3_1_8B_Instruct_v2 import generate_response
print("After import")  # Không load

generate_response("test")
# → "Loading model..." (load ở đây)
```

---

## 🚀 Cách chạy lại

```bash
# Stop tất cả chương trình đang chạy (Ctrl+C)

# Chạy lại rag_query.py
uv run rag_query.py
```

**Output mới:**
```
🔄 Loading LLM (GPU priority)...
   ✅ Llama 3.1 8B loaded  ← Chưa load model, chỉ import

🔄 Loading index from disk...
   ✅ Loaded 88 chunks

🔄 Loading embedding & reranker (CPU)...
   ✅ Embedding model loaded (CPU)
   ✅ Reranker loaded (CPU)

💬 Câu hỏi: NLP là gì?

🤖 Đang xử lý...
Loading model...  ← Load LÚC NÀY (khi query)
Model loaded!     ← Chiếm GPU
   🔍 Searching...
   🎯 Reranking...
   🤖 Generating answer... (5s) ← GPU NHANH!
```

---

## ✅ Kết luận

**Vấn đề:** Model load ngay khi import → Không kiểm soát được thời điểm load

**Giải pháp:** Lazy loading → Load khi gọi function lần đầu

**Kết quả:** LLM load đúng lúc, chiếm GPU, chạy nhanh! 🚀

---

## 📝 Pattern: Lazy Loading

**Áp dụng cho mọi model nặng:**

```python
# Global variables
_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    
    # Load chỉ 1 lần
    _model = load_heavy_model()
    return _model

def use_model(...):
    model = _load_model()  # Load khi cần
    return model.predict(...)
```

**Lợi ích:**
- ✅ Kiểm soát thời điểm load
- ✅ Tránh load không cần thiết
- ✅ Dễ debug
- ✅ Tiết kiệm VRAM
