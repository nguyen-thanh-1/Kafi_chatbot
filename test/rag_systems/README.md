# RAG Systems

Thư mục chứa các RAG system implementations.

## Structure

```
rag_systems/
├── rag_pro/          # RAG Pro V1 & V2 (optimized)
├── qwen_rag/         # Qwen-based RAG
├── lightrag/         # LightRAG integration
└── api_rag/          # API-based RAG (Gemini, Groq)
```

## RAG Pro (Recommended)

**Location:** `rag_pro/`

**Files:**
- `rag_pro_v2.py` - Optimized RAG with OOM fixes ✅
- `rag_index.py` - Index-only script (GPU embedding)
- `rag_query.py` - Query-only script (GPU LLM)
- `rag_config.py` - Configuration
- `rag_pro.py` - V1 (legacy)

**Usage:**
```bash
# Index documents
cd rag_systems/rag_pro
uv run rag_index.py --force

# Query
uv run rag_query.py
```

**Features:**
- Semantic chunking (87% fewer chunks)
- Batch embedding (3-5x faster)
- Embedding cache (1000x faster re-runs)
- FAISS IVF index (5-10x faster search)
- Two-stage retrieval (better quality)
- OOM fixes (hard token limit, smaller chunks)

---

## Qwen RAG

**Location:** `qwen_rag/`

**Files:**
- `Qwen2.5_14B_RAG.py` - Basic Qwen RAG
- `Qwen2.5_14B_RAG_Pro.py` - Optimized Qwen RAG

---

## LightRAG

**Location:** `lightrag/`

**Files:**
- `index_lightrag.py` - Index with LightRAG
- `query_lightrag.py` - Query with LightRAG

---

## API RAG

**Location:** `api_rag/`

**Files:**
- `index_docs_gemini.py` - Index with Gemini API
- `index_docs_groq.py` - Index with Groq API
- `query_rag_gemini.py` - Query with Gemini
- `query_rag_groq.py` - Query with Groq
- `index_docs.py` - Generic indexer
- `query_rag.py` - Generic query

---

## Comparison

| System | Speed | Quality | VRAM | Cost |
|--------|-------|---------|------|------|
| **RAG Pro V2** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 12-14GB | Free |
| Qwen RAG Pro | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 16GB+ | Free |
| LightRAG | ⭐⭐⭐ | ⭐⭐⭐ | 12GB | Free |
| API RAG | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 0GB | Paid |

**Recommended:** RAG Pro V2 (best balance)
