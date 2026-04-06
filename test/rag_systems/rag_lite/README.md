# RAG Lite - Lightweight RAG System

## 📊 Components

| Component | Model | Speed | Size |
|-----------|-------|-------|------|
| **Embedding** | all-MiniLM-L6-v2 | ⭐⭐⭐⭐⭐ | 384 dim |
| **Reranker** | FlashRank (ONNX) | ⭐⭐⭐⭐⭐ | ~50MB |
| **Chunking** | Recursive | ⭐⭐⭐⭐ | 1000 chars |
| **LLM** | Llama 3.1 8B | ⭐⭐⭐⭐ | 4-bit |

---

## 🚀 Quick Start

### 1. Index Documents

```bash
cd rag_systems/rag_lite
uv run rag_index.py --force
```

### 2. Query

```bash
uv run rag_query.py
```

---

## 🔧 Configuration

In `rag_lite.py`:

```python
# Chunking
CHUNK_SIZE = 1000        # characters
CHUNK_OVERLAP = 200      # characters

# Retrieval
TOP_K_RETRIEVE = 20      # FAISS search
TOP_K_RERANK = 3         # FlashRank output

# Context (VRAM safe)
MAX_CONTEXT_TOKENS = 1200
LLM_MAX_TOKENS = 700
```

---

## 📁 File Structure

```
rag_lite/
├── rag_lite.py      # Main implementation
├── rag_index.py     # Index-only script
├── rag_query.py     # Query-only script
└── README.md        # This file
```

---

## ⚡ Performance vs RAG Pro

| Metric | RAG Lite | RAG Pro |
|--------|----------|---------|
| **Embedding** | MiniLM (5x faster) | BGE-M3 |
| **Reranker** | FlashRank (10x faster) | CrossEncoder |
| **Embedding VRAM** | ~0.2GB | ~1GB |
| **Reranker VRAM** | 0GB (ONNX) | ~2GB |
| **Quality** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 Use Cases

- ✅ Quick prototyping
- ✅ Low resource environments
- ✅ Real-time applications
- ✅ High throughput

---

## 📚 Recursive Chunking

```
Strategy:
1. Try split by \n\n (paragraphs)
2. If too large, try \n (lines)
3. If still too large, try ". " (sentences)
4. If still too large, try " " (words)
5. Last resort: character split

Benefits:
- Preserves document structure
- Consistent chunk sizes
- Better context
```

---

## 🔗 Dependencies

```
flashrank
sentence-transformers
faiss-cpu
torch
pdfplumber
```

Install:
```bash
uv add flashrank
```
