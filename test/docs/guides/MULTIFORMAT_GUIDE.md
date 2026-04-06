# PageIndex Multi-Format RAG - Hướng dẫn

## 📌 Giải thích các file

### 1. `pageindex_llama_rag_simple.py`
**Mục đích:** Phiên bản đơn giản để test, import LLM từ file `Llama_3_1_8B_Instruct_v2.py`

**Đặc điểm:**
- Không load lại model, sử dụng model đã có
- Chỉ hỗ trợ file `.txt`
- Dùng để test nhanh khi model đã được load sẵn

**Khi nào dùng:**
- Khi bạn đã có model running
- Test PageIndex logic mà không cần load model lại

### 2. `pageindex_llama_rag.py`
**Mục đích:** Hệ thống RAG hoàn chỉnh, load model trực tiếp

**Đặc điểm:**
- Load Llama 3.1 8B với quantization 4-bit
- Fallback tự động sang FP16
- Chỉ hỗ trợ file `.txt`
- Standalone, không phụ thuộc file khác

**Khi nào dùng:**
- Production use
- Khi muốn control hoàn toàn việc load model

### 3. `pageindex_multiformat.py` ⭐ MỚI
**Mục đích:** Hệ thống RAG hỗ trợ NHIỀU định dạng file

**Đặc điểm:**
- ✅ Hỗ trợ TXT, PDF, DOCX, MD
- ✅ Tự động detect file type
- ✅ Specialized readers cho từng format
- ✅ Intelligent section splitting

**Khi nào dùng:**
- Khi bạn có tài liệu PDF, Word
- Muốn index nhiều loại file cùng lúc

## 🚀 Hỗ trợ định dạng mới

### Hiện tại chỉ hỗ trợ .txt
❌ `pageindex_llama_rag.py` - Chỉ TXT  
❌ `pageindex_llama_rag_simple.py` - Chỉ TXT

### Bây giờ hỗ trợ đa định dạng
✅ `pageindex_multiformat.py` - TXT, PDF, DOCX, MD

## 📦 Cài đặt dependencies

```bash
# Cho PDF support
uv pip install pypdf

# Cho DOCX support  
uv pip install python-docx
```

## 💡 Cách sử dụng Multi-Format

### Bước 1: Thêm tài liệu
Thêm file vào `./courses/`:
```
courses/
├── document1.txt
├── report.pdf
├── thesis.docx
└── notes.md
```

### Bước 2: Chạy hệ thống
```bash
python pageindex_multiformat.py
```

### Bước 3: Hệ thống tự động index
```
📚 Đang xây dựng PageIndex từ ./courses...
  📄 document1 (TXT): 5 sections
  📄 report (PDF): 12 sections
  📄 thesis (DOCX): 8 sections
  📄 notes (MD): 6 sections
✅ Đã index 4 tài liệu với 31 sections
```

## 🔍 Đặc điểm từng format

### TXT Files
- Tách theo đoạn văn (`\n\n`)
- Đơn giản, nhanh

### PDF Files
- Trích xuất text từ mỗi trang
- Giữ thông tin số trang
- Tách theo trang hoặc đoạn văn

### DOCX Files
- Đọc paragraphs từ Word
- Bảo toàn cấu trúc văn bản
- Tách theo đoạn văn

### Markdown Files
- Tách theo headers (`#`, `##`, etc.)
- Bảo toàn cấu trúc phân cấp
- Phù hợp với documentation

## 📊 So sánh 3 files

| Feature | simple.py | rag.py | multiformat.py |
|---------|-----------|--------|----------------|
| Load model | Import từ file | Load trực tiếp | Load trực tiếp |
| TXT | ✅ | ✅ | ✅ |
| PDF | ❌ | ❌ | ✅ |
| DOCX | ❌ | ❌ | ✅ |
| MD | ❌ | ❌ | ✅ |
| Quantization | Phụ thuộc | 4-bit + FP16 | 4-bit + FP16 |
| Use case | Testing | Production | Multi-format |

## 🎯 Khuyến nghị

### Nếu chỉ có file TXT
→ Dùng `pageindex_llama_rag.py`

### Nếu có PDF, DOCX, MD
→ Dùng `pageindex_multiformat.py` ⭐

### Nếu đang test logic
→ Dùng `pageindex_llama_rag_simple.py`

## 📝 Ví dụ sử dụng

### Với file PDF
```python
# Thêm file report.pdf vào ./courses/
# Chạy:
python pageindex_multiformat.py

# Hỏi:
💬 Câu hỏi: Tóm tắt báo cáo này

# Kết quả sẽ trích xuất từ PDF và trả lời
```

### Với file DOCX
```python
# Thêm file thesis.docx vào ./courses/
# Rebuild index:
💬 Câu hỏi: rebuild

# Hỏi:
💬 Câu hỏi: Phương pháp nghiên cứu là gì?
```

## ⚙️ Tùy chỉnh

### Thêm format mới
Chỉnh sửa `DocumentReader` class trong `pageindex_multiformat.py`:

```python
@staticmethod
def read_custom_format(file_path: Path) -> str:
    # Your custom reader
    pass
```

### Thay đổi cách tách sections
Chỉnh sửa `_split_into_sections()` method:

```python
def _split_into_sections(self, content: str, file_type: str):
    if file_type == 'your_format':
        # Custom splitting logic
        pass
```

## 🐛 Troubleshooting

### Lỗi: "pypdf không được cài đặt"
```bash
uv pip install pypdf
```

### Lỗi: "python-docx không được cài đặt"
```bash
uv pip install python-docx
```

### PDF không đọc được
- Kiểm tra PDF có text layer không (không phải scan)
- Một số PDF bảo mật không đọc được

### DOCX lỗi format
- Đảm bảo file DOCX không bị corrupt
- Thử mở bằng Word để verify

## 🎉 Kết luận

Bây giờ bạn có thể:
- ✅ Sử dụng file PDF cho RAG
- ✅ Sử dụng file DOCX cho RAG
- ✅ Sử dụng file Markdown cho RAG
- ✅ Mix nhiều format trong cùng 1 hệ thống
