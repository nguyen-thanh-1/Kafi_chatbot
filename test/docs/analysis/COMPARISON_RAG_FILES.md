# So sánh chi tiết: pageindex_llama_rag.py vs pageindex_llama_rag_simple.py

## 🔍 Tổng quan

Cả 2 file đều là hệ thống PageIndex RAG với Llama 3.1 8B, nhưng có **sự khác biệt quan trọng** về cách load model.

## 📊 Bảng so sánh

| Đặc điểm | `pageindex_llama_rag.py` | `pageindex_llama_rag_simple.py` |
|----------|--------------------------|----------------------------------|
| **Cách load model** | Load trực tiếp từ HuggingFace | Import từ file `Llama_3_1_8B_Instruct_v2.py` |
| **Dependencies** | `torch`, `transformers`, `bitsandbytes` | Phụ thuộc vào file `Llama_3_1_8B_Instruct_v2.py` |
| **Quantization** | 4-bit (BitsAndBytesConfig) + Fallback FP16 | Phụ thuộc vào file được import |
| **Standalone** | ✅ Độc lập hoàn toàn | ❌ Cần file `Llama_3_1_8B_Instruct_v2.py` |
| **LLM Wrapper** | Class `LlamaLLM` riêng | Sử dụng `llama.generate_response()` |
| **PageIndex** | Import từ `pageindex_core.py` | Tự implement class `LocalPageIndex` |
| **Số dòng code** | 289 dòng | 312 dòng |
| **Phù hợp cho** | Production, deployment | Testing, development |

## 🔑 Điểm khác biệt chính

### 1. Cách load Model

#### `pageindex_llama_rag.py` (STANDARD)
```python
class LlamaLLM:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.load_model()  # Load trực tiếp
    
    def load_model(self):
        # Cấu hình quantization 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer và model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )
```

**Ưu điểm:**
- ✅ Độc lập, không phụ thuộc file khác
- ✅ Control đầy đủ việc load model
- ✅ Có fallback tự động sang FP16 nếu 4-bit lỗi
- ✅ Phù hợp production

**Nhược điểm:**
- ❌ Mất thời gian load model (1-2 phút)
- ❌ Cần cấu hình quantization

---

#### `pageindex_llama_rag_simple.py` (SIMPLE)
```python
def import_llm_module():
    """Import module Llama từ file có sẵn"""
    module_path = Path(__file__).parent / "Llama_3_1_8B_Instruct_v2.py"
    
    spec = importlib.util.spec_from_file_location("llama_module", module_path)
    llama_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llama_module)
    
    return llama_module

# Load LLM
llama = import_llm_module()

# Sử dụng
response = llama.generate_response(
    user_input=user_prompt,
    history=history,
    max_new_tokens=max_new_tokens,
    temperature=temperature
)
```

**Ưu điểm:**
- ✅ Nhanh hơn nếu model đã được load trong file khác
- ✅ Tái sử dụng code có sẵn
- ✅ Đơn giản, dễ test

**Nhược điểm:**
- ❌ Phụ thuộc vào file `Llama_3_1_8B_Instruct_v2.py`
- ❌ Không control được cách load model
- ❌ Vẫn phải load model lần đầu (không tiết kiệm thời gian)

### 2. PageIndex Implementation

#### `pageindex_llama_rag.py`
```python
# Import từ module riêng
from pageindex_core import LocalPageIndex, format_context_for_prompt

# Sử dụng
self.page_index = LocalPageIndex(documents_dir)
```

**Ưu điểm:**
- ✅ Code gọn gàng, modular
- ✅ Tái sử dụng được cho nhiều file
- ✅ Dễ maintain

---

#### `pageindex_llama_rag_simple.py`
```python
# Tự implement class LocalPageIndex trong file
class LocalPageIndex:
    def __init__(self, documents_dir="./courses"):
        self.documents_dir = Path(documents_dir)
        self.index = {}
        self.documents = {}
    
    # ... toàn bộ implementation
```

**Ưu điểm:**
- ✅ Standalone, không cần import
- ✅ Dễ đọc toàn bộ logic trong 1 file

**Nhược điểm:**
- ❌ Code dài hơn (duplicate code)
- ❌ Khó maintain khi có nhiều file

### 3. LLM Interface

#### `pageindex_llama_rag.py`
```python
class LlamaLLM:
    def chat(self, messages, max_new_tokens=512, temperature=0.2):
        """Chat với history"""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return self.generate(prompt, max_new_tokens, temperature)

# Sử dụng
response = self.llm.chat(messages, max_new_tokens, temperature)
```

---

#### `pageindex_llama_rag_simple.py`
```python
# Sử dụng function từ file được import
response = llama.generate_response(
    user_input=user_prompt,
    history=history,
    max_new_tokens=max_new_tokens,
    temperature=temperature
)
```

## 🎯 Khi nào dùng file nào?

### Dùng `pageindex_llama_rag.py` khi:
✅ **Production deployment**  
✅ Muốn control đầy đủ việc load model  
✅ Cần quantization 4-bit để tiết kiệm VRAM  
✅ Muốn code modular, dễ maintain  
✅ Không muốn phụ thuộc file khác  

**Ví dụ:**
```bash
# Chạy standalone
python pageindex_llama_rag.py

# Hoặc
uv run pageindex_llama_rag.py
```

---

### Dùng `pageindex_llama_rag_simple.py` khi:
✅ **Testing và development**  
✅ Đã có file `Llama_3_1_8B_Instruct_v2.py` working  
✅ Muốn test PageIndex logic nhanh  
✅ Không quan tâm cách load model  

**Ví dụ:**
```bash
# Cần có file Llama_3_1_8B_Instruct_v2.py
python pageindex_llama_rag_simple.py
```

## 📝 Code Structure Comparison

### `pageindex_llama_rag.py`
```
pageindex_llama_rag.py (289 dòng)
├── Import statements
│   ├── torch, transformers
│   └── pageindex_core (LocalPageIndex, format_context_for_prompt)
├── LlamaLLM class (88 dòng)
│   ├── __init__()
│   ├── load_model() - với quantization + fallback
│   ├── generate()
│   └── chat()
├── PageIndexRAG class (86 dòng)
│   ├── __init__() - khởi tạo PageIndex + LLM
│   ├── query()
│   ├── rebuild_index()
│   └── get_statistics()
└── main() - Interactive interface
```

### `pageindex_llama_rag_simple.py`
```
pageindex_llama_rag_simple.py (312 dòng)
├── Import statements
│   └── importlib.util
├── import_llm_module() - Import từ file
├── LocalPageIndex class (145 dòng) - Tự implement
│   ├── __init__()
│   ├── build_index()
│   ├── _index_document()
│   ├── search()
│   ├── _calculate_relevance()
│   └── get_context()
├── PageIndexRAG class (75 dòng)
│   ├── __init__()
│   ├── query() - dùng llama.generate_response()
│   └── rebuild_index()
└── main() - Interactive interface
```

## 💡 Khuyến nghị

### Cho người mới bắt đầu
→ Dùng **`pageindex_llama_rag.py`**
- Đơn giản hơn, ít phụ thuộc
- Có error handling tốt hơn
- Documentation rõ ràng

### Cho developer có kinh nghiệm
→ Dùng **`pageindex_llama_rag.py`** cho production
→ Dùng **`pageindex_llama_rag_simple.py`** cho testing

### Cho multi-format (PDF, DOCX)
→ Dùng **`pageindex_multiformat.py`** ⭐
- Hỗ trợ nhiều format nhất
- Architecture tốt nhất
- Recommended!

## 🔄 Migration Path

Nếu bạn đang dùng `pageindex_llama_rag_simple.py` và muốn chuyển sang `pageindex_llama_rag.py`:

```bash
# Không cần thay đổi gì
# Chỉ cần chạy file mới
python pageindex_llama_rag.py

# Model sẽ được load tự động
# Tất cả chức năng giống nhau
```

## 📊 Performance Comparison

| Metric | Standard | Simple |
|--------|----------|--------|
| Startup time | ~60-120s (load model) | ~60-120s (load model) |
| Memory usage | ~6GB VRAM (4-bit) | Phụ thuộc file import |
| Query speed | Giống nhau | Giống nhau |
| Code maintainability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Flexibility | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## ✅ Kết luận

**TL;DR:**
- `pageindex_llama_rag.py` = **Production-ready**, standalone, modular
- `pageindex_llama_rag_simple.py` = **Testing**, phụ thuộc file khác, monolithic

**Khuyến nghị:** Dùng `pageindex_llama_rag.py` hoặc `pageindex_multiformat.py` cho hầu hết use cases.
