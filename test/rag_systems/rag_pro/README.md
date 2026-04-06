# RAG Pro - Quick Start Guide

## 📍 Location

```
rag_systems/rag_pro/
├── rag_pro_v2.py      # Main RAG implementation
├── rag_index.py       # Index-only script
├── rag_query.py       # Query-only script
└── rag_config.py      # Configuration
```

---

## 🚀 Quick Start

### 1. Index Documents

```bash
# Navigate to rag_pro folder
cd rag_systems/rag_pro

# Index documents (first time or force rebuild)
uv run rag_index.py --force

# Or from project root
uv run rag_systems/rag_pro/rag_index.py --force
```

**What it does:**
- Loads documents from `data/courses/`
- Chunks text (400-800 words)
- Embeds chunks using BGE-M3 (GPU)
- Saves to `storage/rag_storage_pro_v2/`

**Time:** ~6-10 min for 800-page PDF (first time)

---

### 2. Query

```bash
# Navigate to rag_pro folder
cd rag_systems/rag_pro

# Start interactive query
uv run rag_query.py

# Or from project root
uv run rag_systems/rag_pro/rag_query.py
```

**What it does:**
- Loads index from disk
- Embeds query (CPU)
- Searches FAISS index (50 chunks)
- Reranks to top 3 (CPU)
- Generates answer with LLM (GPU)

**Time:** ~6-7 sec per query

---

## 📂 File Paths Reference

### Project Structure
```
Special_Subject_AI/                    # Project root
├── llm_models/                        # LLM wrappers
│   └── Llama_3_1_8B_Instruct_v2.py   # Used by rag_pro
├── rag_systems/
│   └── rag_pro/                       # ← You are here
│       ├── rag_pro_v2.py             # Main implementation
│       ├── rag_index.py              # Index script
│       ├── rag_query.py              # Query script
│       └── rag_config.py             # Config
├── data/
│   └── courses/                       # Input documents
└── storage/
    └── rag_storage_pro_v2/           # Index storage
        ├── faiss_index.bin           # FAISS index
        ├── chunks.pkl                # Chunk data
        ├── indexed_files.json        # File tracker
        └── embedding_cache.pkl       # Cache
```

### Absolute Paths

**Scripts:**
- `C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\rag_pro\rag_index.py`
- `C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\rag_pro\rag_query.py`
- `C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\rag_pro\rag_pro_v2.py`

**Dependencies:**
- LLM: `C:\Users\Admin\Desktop\Special_Subject_AI\llm_models\Llama_3_1_8B_Instruct_v2.py`

**Data:**
- Input: `C:\Users\Admin\Desktop\Special_Subject_AI\data\courses\`
- Storage: `C:\Users\Admin\Desktop\Special_Subject_AI\storage\rag_storage_pro_v2\`

---

## ⚙️ Configuration

Edit `rag_config.py` or modify constants in `rag_pro_v2.py`:

```python
# Chunking
MIN_CHUNK_SIZE = 400   # words
MAX_CHUNK_SIZE = 800   # words
CHUNK_OVERLAP = 100    # words

# Retrieval
TOP_K_RETRIEVE = 50    # FAISS search
TOP_K_RERANK = 3       # Reranker output

# Context (OOM prevention)
MAX_CONTEXT_TOKENS = 2000  # Hard limit

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
```

---

## 🎯 Usage Examples

### Example 1: Index from Different Folder

```bash
# From project root
cd C:\Users\Admin\Desktop\Special_Subject_AI
uv run rag_systems/rag_pro/rag_index.py --force
```

### Example 2: Query from Different Folder

```bash
# From project root
cd C:\Users\Admin\Desktop\Special_Subject_AI
uv run rag_systems/rag_pro/rag_query.py
```

### Example 3: Single Query

```bash
cd rag_systems/rag_pro
uv run rag_query.py --query "What is NLP?"
```

---

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'Llama_3_1_8B_Instruct_v2'
```

**Solution:**
The script automatically adds project root to `sys.path`. If this fails:

```python
# Add to top of script
import sys
import os
project_root = r"C:\Users\Admin\Desktop\Special_Subject_AI"
sys.path.insert(0, project_root)
```

### Issue: Index not found

**Error:**
```
❌ Lỗi: Index not found
💡 Hãy chạy rag_index.py trước để tạo index!
```

**Solution:**
```bash
cd rag_systems/rag_pro
uv run rag_index.py --force
```

### Issue: CUDA OOM

**Error:**
```
torch.cuda.OutOfMemoryError
```

**Solution:**
Already fixed! The code uses:
- Smaller chunks (400-800 words)
- Hard token limit (2000 tokens)
- TOP_K_RERANK = 3

If still OOM, reduce `MAX_CONTEXT_TOKENS` in `rag_pro_v2.py`:
```python
MAX_CONTEXT_TOKENS = 1500  # From 2000
```

---

## 📊 Performance

### Indexing (rag_index.py)
```
Documents: 800-page PDF
Chunks: ~4,000 (semantic chunking)
Time: 6-10 min (first time)
Time: 2-3 sec (with cache)
VRAM: ~3GB (embedding on GPU)
```

### Querying (rag_query.py)
```
Pipeline:
  1. Embed query (CPU): 0.05s
  2. FAISS search: 0.5s
  3. Rerank (CPU): 1.5s
  4. LLM generate (GPU): 5s
  ─────────────────────────
  Total: ~7s

VRAM: ~13.5GB (LLM 12GB + KV cache 1.5GB)
```

---

## 📚 Related Documentation

- **Main Guide:** `docs/guides/RAG_PRO_V2_QUICKSTART.md`
- **OOM Fix:** `docs/fixes/QUERY_OOM_FIX.md`
- **V1 vs V2:** `docs/guides/RAG_PRO_V1_VS_V2.md`
- **Split Index/Query:** `docs/guides/SPLIT_INDEX_QUERY_GUIDE.md`

---

## ✅ Checklist

Before running:
- [ ] Documents in `data/courses/`
- [ ] GPU available (for LLM)
- [ ] ~16GB VRAM (recommended)
- [ ] Python packages installed (`uv sync`)

First time:
- [ ] Run `rag_index.py --force`
- [ ] Wait for indexing to complete
- [ ] Run `rag_query.py`

---

**Ready to use!** 🚀
