# PageIndex Multi-Format với Gemini API

## ✅ Đã cập nhật

File `pageindex_multiformat.py` đã được cập nhật để sử dụng **Gemini 2.0 Flash Exp** thay vì Llama 3.1 8B.

## 🔧 Thay đổi chính

### 1. LLM Engine
```python
# OLD: Llama 3.1 8B (local)
class LlamaLLM:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        # Load local model với quantization
        
# NEW: Gemini 2.0 Flash Exp (API)
class GeminiLLM:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash-exp"):
        # Kết nối Gemini API
```

### 2. Dependencies
```python
# OLD
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# NEW
import google.generativeai as genai
```

### 3. RAG System
```python
# OLD
rag = MultiFormatRAG(documents_dir="./courses", model_id="meta-llama/...")

# NEW
rag = MultiFormatRAG(
    documents_dir="./courses",
    api_key="YOUR_API_KEY",  # hoặc dùng env var
    model_name="gemini-2.0-flash-exp"
)
```

## 🚀 Cách sử dụng

### Bước 1: Cài đặt dependencies
```bash
uv pip install google-generativeai pypdf python-docx
```

### Bước 2: Set API Key

**Option A: Environment Variable (Khuyến nghị)**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"
```

**Option B: Truyền trực tiếp trong code**
```python
rag = MultiFormatRAG(
    documents_dir="./courses",
    api_key="your-api-key-here"
)
```

### Bước 3: Chạy
```bash
python pageindex_multiformat.py
```

## 📊 Output mới

```
══════════════════════════════════════════════════════════════════════
🚀 PageIndex Multi-Format RAG System (Gemini API)
══════════════════════════════════════════════════════════════════════

📌 Hỗ trợ định dạng:
  ✅ TXT - Text files
  ✅ PDF - PDF documents
  ✅ DOCX - Word documents
  ✅ MD - Markdown files

🤖 LLM: Gemini 2.0 Flash Exp (API)
══════════════════════════════════════════════════════════════════════

📚 Đang xây dựng PageIndex từ ./courses...
  📄 sample_knowledge (TXT): 1 sections
  📄 nlp-book (PDF): 150 sections
✅ Đã index 2 tài liệu với 151 sections

🔄 Đang kết nối Gemini API (gemini-2.0-flash-exp)...
✅ Gemini API sẵn sàng!

📊 Thống kê:
  • Tổng tài liệu: 2
  • Tổng sections: 151
  • Theo loại:
    - TXT: 1 files
    - PDF: 1 files

✅ Hệ thống sẵn sàng!

📝 Lệnh: rebuild | stats | exit
══════════════════════════════════════════════════════════════════════

💬 Câu hỏi: 
```

## 💡 Ưu điểm của Gemini API

### So với Llama 3.1 8B Local:

| Tiêu chí | Llama 3.1 8B | Gemini 2.0 Flash |
|----------|--------------|------------------|
| **VRAM** | 12-16GB | 0GB (API) |
| **Setup** | Phức tạp (quantization) | Đơn giản (API key) |
| **Speed** | Phụ thuộc GPU | Nhanh (Google infra) |
| **Quality** | Tốt | Rất tốt |
| **Cost** | Free (local) | Pay-per-use |
| **Maintenance** | Tự quản lý | Google quản lý |

### ✅ Pros:
- Không cần GPU/VRAM
- Setup đơn giản (chỉ cần API key)
- Nhanh và ổn định
- Chất lượng cao
- Không lo CUDA OOM

### ⚠️ Cons:
- Cần internet
- Có chi phí (nhưng rất rẻ)
- Phụ thuộc Google API

## 📝 Example Usage

```python
from pageindex_multiformat import MultiFormatRAG

# Khởi tạo với Gemini API
rag = MultiFormatRAG(
    documents_dir="./courses",
    api_key="your-api-key-here",  # hoặc dùng env var
    model_name="gemini-2.0-flash-exp"
)

# Query
response, sources = rag.query("Machine Learning là gì?")
print(response)
print("Nguồn:", sources)
```

## 🔑 Lấy Gemini API Key

1. Truy cập: https://aistudio.google.com/apikey
2. Đăng nhập Google account
3. Click "Create API Key"
4. Copy API key
5. Set environment variable hoặc truyền vào code

## ⚡ Performance

### Gemini 2.0 Flash Exp:
- **Speed:** ~1-2 giây/response
- **Quality:** Rất tốt (comparable với GPT-4)
- **Cost:** ~$0.00001/1K tokens (rất rẻ)
- **Rate Limit:** 15 RPM (free tier)

### So với Llama local:
- **Nhanh hơn** nếu không có GPU mạnh
- **Chất lượng tốt hơn** trong nhiều task
- **Dễ setup hơn** (không cần GPU)

## 🎯 Kết luận

**Gemini API là lựa chọn tốt hơn cho:**
- Máy không có GPU mạnh
- Muốn setup nhanh
- Cần chất lượng cao
- Không muốn lo CUDA OOM

**Llama local tốt hơn cho:**
- Có GPU mạnh (16GB+ VRAM)
- Cần privacy tuyệt đối
- Không muốn phụ thuộc internet
- Không muốn trả phí

---

**Bây giờ hãy thử chạy với Gemini API!** 🚀
