# RAG Evaluation với Opik

Đánh giá 2 hệ thống RAG (`rag_llama.py` và `rag_qwen3_8b.py`) bằng [Opik](https://www.comet.com/opik) — **hoàn toàn local, không dùng API bên ngoài** để generate.

## Cấu trúc

```
optimize_rag/
├── rag_llama.py            ← RAG với LLaMA 3.1-8B-Instruct
├── rag_qwen3_8b.py         ← RAG với Qwen3-8B
├── test_dataset.json       ← 100 câu hỏi NLP Q&A
├── eval_rag_llama.py       ← ← Eval script cho LLaMA   (project: rag_llama)
└── eval_rag_qwen3_8b.py    ← ← Eval script cho Qwen3   (project: rag_qwen3_8b)
```

## Metrics sử dụng

| Metric | Loại | Mô tả |
|---|---|---|
| **LevenshteinRatio** | Heuristic | Độ tương đồng ký tự giữa answer và expected |
| **ROUGE-L** | Heuristic | Overlap n-gram dài nhất chung (LCS) |

> Không dùng LLM-as-judge → không cần API bên ngoài.

## Bước 1 — Lấy OPIK API Key

1. Vào https://www.comet.com/opik/thanh-nguy-n-1461/projects
2. Click avatar góc trên phải → **API Keys** → Copy key
3. Đặt vào `.env`:
   ```
   OPIK_API_KEY=your_opik_api_key_here
   ```

## Bước 2 — Cài dependencies

```powershell
uv sync
```

## Bước 3 — Chạy eval

> ⚠️ **Quan trọng**: Chạy TỪNG CÁI MỘT vì 2 LLM dùng hết VRAM — không chạy song song.

### Eval rag_llama (project: `rag_llama`)
```powershell
# Từ thư mục gốc Special_Subject_AI
uv run python rag_systems/optimize_rag/eval_rag_llama.py
```

### Eval rag_qwen3_8b (project: `rag_qwen3_8b`)  
```powershell
# Sau khi eval_rag_llama.py chạy xong
uv run python rag_systems/optimize_rag/eval_rag_qwen3_8b.py
```

## Bước 4 — Xem kết quả

Truy cập: https://www.comet.com/opik/thanh-nguy-n-1461/projects

Sẽ thấy 2 project:
- **rag_llama** — kết quả của LLaMA 3.1-8B-Instruct
- **rag_qwen3_8b** — kết quả của Qwen3-8B

Mỗi experiment lưu:
- Câu hỏi, câu trả lời generated, expected output
- Context được retrieved
- Scores: LevenshteinRatio + ROUGE-L
- Metadata: retrieval_time, generation_time, num_contexts, model info

## Lưu ý

- Script tự động **load FAISS index** nếu đã có sẵn (`faiss.index` / `faiss_qwen3_8b.index`)
- Nếu chưa có index → tự **build từ PDF** (`nlp-book.pdf`)
- Mỗi lần chạy tạo 1 **experiment mới** (timestamp trong tên) → có thể so sánh nhiều lần chạy
