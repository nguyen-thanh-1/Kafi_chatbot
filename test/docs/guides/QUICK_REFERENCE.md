# Quick Reference: Chọn file PageIndex RAG nào?

## 🎯 Decision Tree

```
Bạn cần gì?
│
├─ Chỉ file TXT?
│  │
│  ├─ Production → pageindex_llama_rag.py ✅
│  └─ Testing → pageindex_llama_rag_simple.py
│
└─ Có PDF/DOCX/MD?
   │
   └─ → pageindex_multiformat.py ⭐ (RECOMMENDED)
```

## 📁 3 Files chính

### 1️⃣ `pageindex_llama_rag.py` - STANDARD
```
┌─────────────────────────────────────┐
│  PageIndex RAG (Standard)           │
├─────────────────────────────────────┤
│ ✅ Load model trực tiếp             │
│ ✅ Quantization 4-bit + FP16        │
│ ✅ Standalone (không phụ thuộc)     │
│ ✅ Modular architecture             │
│ ❌ Chỉ TXT                          │
├─────────────────────────────────────┤
│ Use case: Production, TXT only      │
└─────────────────────────────────────┘
```

### 2️⃣ `pageindex_llama_rag_simple.py` - SIMPLE
```
┌─────────────────────────────────────┐
│  PageIndex RAG (Simple)             │
├─────────────────────────────────────┤
│ ✅ Import từ file có sẵn            │
│ ✅ Đơn giản, dễ test                │
│ ❌ Phụ thuộc Llama_3_1_8B_v2.py     │
│ ❌ Monolithic code                  │
│ ❌ Chỉ TXT                          │
├─────────────────────────────────────┤
│ Use case: Testing, Development      │
└─────────────────────────────────────┘
```

### 3️⃣ `pageindex_multiformat.py` - MULTI-FORMAT ⭐
```
┌─────────────────────────────────────┐
│  PageIndex RAG (Multi-Format)       │
├─────────────────────────────────────┤
│ ✅ Load model trực tiếp             │
│ ✅ Quantization 4-bit + FP16        │
│ ✅ Standalone                       │
│ ✅ TXT, PDF, DOCX, MD               │
│ ✅ Specialized readers              │
├─────────────────────────────────────┤
│ Use case: Production, Multi-format  │
│ RECOMMENDED! 🌟                     │
└─────────────────────────────────────┘
```

## 🔑 Key Differences

### Model Loading

| File | Method | Code |
|------|--------|------|
| **standard** | Direct load | `LlamaLLM(model_id)` |
| **simple** | Import file | `import_llm_module()` |
| **multiformat** | Direct load | `LlamaLLM(model_id)` |

### File Support

| File | TXT | PDF | DOCX | MD |
|------|-----|-----|------|-----|
| **standard** | ✅ | ❌ | ❌ | ❌ |
| **simple** | ✅ | ❌ | ❌ | ❌ |
| **multiformat** | ✅ | ✅ | ✅ | ✅ |

### Dependencies

| File | External Files | Packages |
|------|----------------|----------|
| **standard** | `pageindex_core.py` | torch, transformers |
| **simple** | `Llama_3_1_8B_v2.py` | torch, transformers |
| **multiformat** | `pageindex_core.py` | torch, transformers, pypdf, python-docx |

## 💡 Quick Tips

### Nếu bạn có file PDF
```bash
# Dùng multiformat
uv run pageindex_multiformat.py
```

### Nếu chỉ có file TXT
```bash
# Dùng standard
uv run pageindex_llama_rag.py
```

### Nếu đang test
```bash
# Dùng simple (nếu đã có Llama_3_1_8B_v2.py)
python pageindex_llama_rag_simple.py
```

## 📊 Comparison Matrix

|  | Standard | Simple | Multi-Format |
|---|----------|--------|--------------|
| **Độc lập** | ✅ | ❌ | ✅ |
| **Production** | ✅ | ❌ | ✅ |
| **Multi-format** | ❌ | ❌ | ✅ |
| **Dễ test** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Maintainable** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Flexible** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🎯 Recommendations

### 🥇 Best Choice (Overall)
→ **`pageindex_multiformat.py`**
- Hỗ trợ nhiều format nhất
- Production-ready
- Future-proof

### 🥈 Best Choice (TXT only)
→ **`pageindex_llama_rag.py`**
- Đơn giản, hiệu quả
- Không cần dependencies thừa

### 🥉 Best Choice (Testing)
→ **`pageindex_llama_rag_simple.py`**
- Nhanh để test logic
- Tái sử dụng code có sẵn

## 🚀 Getting Started

### Với Multi-Format (Recommended)
```bash
# 1. Cài đặt
uv pip install pypdf python-docx

# 2. Thêm files vào ./courses/
# (TXT, PDF, DOCX, MD)

# 3. Chạy
uv run pageindex_multiformat.py
```

### Với Standard (TXT only)
```bash
# 1. Thêm files .txt vào ./courses/

# 2. Chạy
uv run pageindex_llama_rag.py
```

## ❓ FAQ

**Q: File nào nhanh nhất?**  
A: Tốc độ query giống nhau. Startup time phụ thuộc vào model loading.

**Q: File nào tốn ít VRAM nhất?**  
A: Cả 3 đều dùng 4-bit quantization → ~6GB VRAM

**Q: Tôi nên dùng file nào?**  
A: 
- Có PDF/DOCX → `pageindex_multiformat.py` ⭐
- Chỉ TXT → `pageindex_llama_rag.py`
- Testing → `pageindex_llama_rag_simple.py`

**Q: Có thể dùng cả 3 files không?**  
A: Có, nhưng không cần thiết. Chọn 1 file phù hợp nhất.

## 📚 Documentation

- `COMPARISON_RAG_FILES.md` - So sánh chi tiết
- `MULTIFORMAT_GUIDE.md` - Hướng dẫn multi-format
- `README_PageIndex.md` - Tài liệu PageIndex
- `QUICKSTART_PageIndex.md` - Quick start

---

**TL;DR:** Dùng `pageindex_multiformat.py` cho hầu hết use cases! 🌟
