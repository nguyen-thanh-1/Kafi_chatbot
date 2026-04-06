# PageIndex + Llama 3.1 8B RAG System

## Giới thiệu

Hệ thống RAG (Retrieval-Augmented Generation) kết hợp phương pháp PageIndex với mô hình Llama 3.1 8B để trả lời câu hỏi dựa trên tài liệu.

### Đặc điểm của PageIndex

PageIndex là một phương pháp RAG **không sử dụng vector** (vectorless), khác biệt với các hệ thống RAG truyền thống:

- ✅ **Không cần vector database**: Sử dụng cấu trúc cây phân cấp thay vì embedding vectors
- ✅ **Không chunking tùy ý**: Tổ chức tài liệu theo cấu trúc tự nhiên (sections)
- ✅ **Reasoning-based retrieval**: Sử dụng LLM để suy luận và tìm kiếm thông tin
- ✅ **Bảo toàn ngữ cảnh**: Giữ nguyên cấu trúc phân cấp của tài liệu
- ✅ **Hoàn toàn local**: Không cần API key bên ngoài

## Cài đặt

### 1. Cài đặt PageIndex

```bash
uv pip install pageindex
```

### 2. Cài đặt các dependencies khác (nếu chưa có)

```bash
uv pip install torch transformers bitsandbytes accelerate
```

## Cấu trúc thư mục

```
Special_Subject_AI/
├── pageindex_llama_rag.py          # File chính
├── Llama_3_1_8B_Instruct_v2.py     # LLM gốc
├── courses/                         # Thư mục chứa tài liệu
│   ├── sample_knowledge.txt
│   └── ... (các file .txt khác)
└── README_PageIndex.md              # File này
```

## Cách sử dụng

### 1. Chạy chương trình

```bash
python pageindex_llama_rag.py
```

### 2. Tương tác với hệ thống

```
Câu hỏi của bạn: Machine Learning là gì?

🤖 Trả lời:
--------------------------------------------------
Machine Learning là một nhánh của trí tuệ nhân tạo...
--------------------------------------------------
```

### 3. Lệnh đặc biệt

- `rebuild`: Xây dựng lại index khi thêm tài liệu mới
- `exit` hoặc `quit`: Thoát chương trình

## Cách thêm tài liệu mới

1. Thêm file `.txt` vào thư mục `./courses/`
2. Chạy lệnh `rebuild` trong chương trình
3. Hoặc khởi động lại chương trình (tự động index)

### Định dạng tài liệu khuyến nghị

```
Tiêu đề chính:
Nội dung của section 1...

Tiêu đề phụ 1:
Nội dung của section 2...

Tiêu đề phụ 2:
Nội dung của section 3...
```

## Kiến trúc hệ thống

### 1. LocalPageIndex Class

Triển khai phương pháp PageIndex ở local:

- **build_index()**: Xây dựng cấu trúc cây phân cấp từ tài liệu
- **search()**: Tìm kiếm sections liên quan dựa trên query
- **get_context()**: Lấy context được format cho LLM

### 2. PageIndexRAG Class

Kết hợp PageIndex với Llama 3.1 8B:

- **query()**: Xử lý câu hỏi và sinh câu trả lời
- **rebuild_index()**: Xây dựng lại index

### 3. Quy trình hoạt động

```
User Query
    ↓
PageIndex Search (Tree-based)
    ↓
Retrieve Relevant Sections
    ↓
Build Context
    ↓
LLM (Llama 3.1 8B) + Context
    ↓
Generate Answer
```

## So sánh với RAG truyền thống

| Đặc điểm | RAG truyền thống | PageIndex RAG |
|----------|------------------|---------------|
| Indexing | Vector embeddings | Tree structure |
| Chunking | Fixed-size chunks | Natural sections |
| Retrieval | Vector similarity | LLM reasoning |
| Context | Arbitrary chunks | Hierarchical sections |
| Explainability | Khó giải thích | Dễ trace |

## Tùy chỉnh

### Thay đổi số lượng sections được retrieve

```python
rag = PageIndexRAG(documents_dir="./courses")
response = rag.query(question, max_sections=5)  # Mặc định: 3
```

### Thay đổi tham số generation

```python
response = rag.query(
    question,
    max_new_tokens=1024,  # Mặc định: 512
    temperature=0.5       # Mặc định: 0.3
)
```

### Thay đổi system prompt

Chỉnh sửa trong class `PageIndexRAG.__init__()`:

```python
self.system_prompt = """Prompt tùy chỉnh của bạn..."""
```

## Ưu điểm

1. **Không cần vector database**: Giảm độ phức tạp và dependencies
2. **Bảo toàn cấu trúc**: Giữ nguyên hierarchy của tài liệu
3. **Dễ debug**: Có thể trace được sections nào được sử dụng
4. **Phù hợp với tài liệu phức tạp**: Báo cáo, sách giáo khoa, tài liệu pháp lý
5. **Hoàn toàn local**: Không cần API key, bảo mật dữ liệu

## Hạn chế

1. **Tốc độ**: Chậm hơn vector search với tài liệu lớn
2. **Scalability**: Khó scale với hàng triệu documents
3. **Phụ thuộc LLM**: Chất lượng retrieval phụ thuộc vào LLM

## Mở rộng

### 1. Sử dụng PageIndex Cloud API

Nếu muốn dùng PageIndex cloud service:

```python
from pageindex import PageIndexClient

pi_client = PageIndexClient(api_key="YOUR_API_KEY")
# Xem docs: https://pageindex.ai/docs
```

### 2. Kết hợp với Embedding

Có thể kết hợp tree-based search với vector search để tăng độ chính xác.

### 3. Multi-modal

Mở rộng để hỗ trợ PDF, images, tables...

## Troubleshooting

### Lỗi: "No documents found"

- Kiểm tra thư mục `./courses/` có file `.txt` không
- Chạy lệnh `rebuild` để xây dựng lại index

### Lỗi: "CUDA out of memory"

- Giảm `max_new_tokens`
- Model đã được quantize 4-bit, nếu vẫn lỗi thì cần GPU lớn hơn

### Kết quả không chính xác

- Thêm nhiều tài liệu liên quan hơn
- Cải thiện cấu trúc tài liệu (tiêu đề rõ ràng)
- Tăng `max_sections` trong query

## Tài liệu tham khảo

- [PageIndex Official Docs](https://pageindex.ai/docs)
- [PageIndex GitHub](https://github.com/pageindex-ai/pageindex)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## License

MIT License
