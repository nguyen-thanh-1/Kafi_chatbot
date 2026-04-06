# 🚀 Quick Start Guide - PageIndex RAG System

## Cài đặt

Package `pageindex` đã được cài đặt:
```bash
uv pip install pageindex  # ✅ Đã hoàn thành
```

## Files đã tạo

### 1. `pageindex_core.py` 
Module PageIndex độc lập - Test và demo cơ chế tree-structured indexing

**Chạy:**
```bash
python pageindex_core.py
```

### 2. `pageindex_llama_rag.py`
Hệ thống RAG hoàn chỉnh với Llama 3.1 8B

**Chạy:**
```bash
python pageindex_llama_rag.py
```

### 3. `README_PageIndex.md`
Tài liệu chi tiết về PageIndex methodology và cách sử dụng

## Cách sử dụng nhanh

### Bước 1: Chuẩn bị tài liệu
Thêm file `.txt` vào thư mục `./courses/`

### Bước 2: Chạy hệ thống
```bash
python pageindex_llama_rag.py
```

### Bước 3: Hỏi đáp
```
💬 Câu hỏi của bạn: Machine Learning là gì?
```

## Lệnh đặc biệt

- `rebuild` - Xây dựng lại index khi thêm tài liệu mới
- `stats` - Xem thống kê hệ thống
- `exit` - Thoát

## Đặc điểm PageIndex

✅ **Vectorless** - Không dùng vector database  
✅ **Tree-structured** - Cấu trúc phân cấp tự nhiên  
✅ **Reasoning-based** - LLM-powered retrieval  
✅ **Context-preserving** - Giữ nguyên hierarchy tài liệu

## Yêu cầu hệ thống

- GPU: Tối thiểu 6GB VRAM (cho Llama 3.1 8B 4-bit)
- Model: Llama 3.1 8B đã download
- Tài liệu: File .txt trong `./courses/`

## Troubleshooting

### Model load chậm
- Lần đầu tiên load model sẽ mất 1-2 phút
- Model được quantize 4-bit để tiết kiệm VRAM

### Không tìm thấy tài liệu
- Kiểm tra thư mục `./courses/` có file `.txt`
- Chạy lệnh `rebuild` trong chương trình

### Lỗi CUDA/bitsandbytes
- Code có fallback tự động sang FP16 nếu 4-bit lỗi
- Nếu vẫn lỗi, kiểm tra GPU driver và CUDA version
