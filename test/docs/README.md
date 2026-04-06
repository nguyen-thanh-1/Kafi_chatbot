# Documentation

Thư mục chứa tất cả documentation.

## Structure

```
docs/
├── guides/      # User guides & tutorials
├── fixes/       # Bug fixes & solutions
└── analysis/    # Technical analysis
```

## Guides

**Location:** `guides/`

- `RAG_PRO_V1_VS_V2.md` - Comparison between V1 and V2
- `RAG_PRO_V2_QUICKSTART.md` - Quick start guide for V2
- `PAGEINDEX_GEMINI_GUIDE.md` - PageIndex with Gemini API
- `MULTIFORMAT_GUIDE.md` - Multi-format support guide
- `SPLIT_INDEX_QUERY_GUIDE.md` - Separated index/query guide
- `QUICK_REFERENCE.md` - Quick reference
- `QUICKSTART_PageIndex.md` - PageIndex quick start
- `SUMMARY_MULTIFORMAT.md` - Multi-format summary

## Fixes

**Location:** `fixes/`

- `CUDA_OOM_FIX.md` - Fix for CUDA out of memory (embedding/reranker CPU)
- `QUERY_OOM_FIX.md` - Fix for query OOM (token limit, smaller chunks)
- `FIX_LAZY_LOADING.md` - Lazy loading implementation
- `FIX_LLM_CPU_ISSUE.md` - Fix LLM running on CPU
- `RERANKER_GPU_UPDATE.md` - Reranker GPU update

## Analysis

**Location:** `analysis/`

- `COMPARISON_RAG_FILES.md` - RAG files comparison
- `SPLIT_INDEX_QUERY_ANALYSIS.md` - Index/query separation analysis
- `SET_GEMINI_KEY.md` - Gemini API key setup

---

## Quick Links

### Getting Started
1. [RAG Pro V2 Quick Start](guides/RAG_PRO_V2_QUICKSTART.md)
2. [PageIndex Quick Start](guides/QUICKSTART_PageIndex.md)

### Common Issues
1. [CUDA OOM Fix](fixes/CUDA_OOM_FIX.md)
2. [Query OOM Fix](fixes/QUERY_OOM_FIX.md)
3. [LLM CPU Issue](fixes/FIX_LLM_CPU_ISSUE.md)

### Advanced
1. [V1 vs V2 Comparison](guides/RAG_PRO_V1_VS_V2.md)
2. [Split Index/Query](guides/SPLIT_INDEX_QUERY_GUIDE.md)
