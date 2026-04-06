# Import Path Fix - After Reorganization

## 🔴 Problem

After reorganization, import paths were broken:

```python
# Old (when files in root)
from Llama_3_1_8B_Instruct_v2 import generate_response

# Error after reorganization
ModuleNotFoundError: No module named 'Llama_3_1_8B_Instruct_v2'
```

---

## ✅ Solution

Add project root to `sys.path` before importing:

```python
def get_llm():
    global _llm
    if _llm is None:
        # Add project root to path
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Now can import from llm_models
        from llm_models.Llama_3_1_8B_Instruct_v2 import generate_response
        _llm = generate_response
    return _llm
```

---

## 📂 Path Calculation

From `rag_systems/rag_pro/rag_pro_v2.py`:

```python
__file__                    # rag_systems/rag_pro/rag_pro_v2.py
os.path.dirname(__file__)   # rag_systems/rag_pro
'..'                        # rag_systems
'..', '..'                  # Special_Subject_AI (project root)
```

---

## 🎯 Files Fixed

- `rag_systems/rag_pro/rag_pro_v2.py` ✅
- `rag_systems/rag_pro/rag_index.py` (uses rag_pro_v2) ✅
- `rag_systems/rag_pro/rag_query.py` (uses rag_pro_v2) ✅

---

## 📚 README Files Created

Each folder now has detailed README with exact paths:

1. **`llm_models/README.md`**
   - LLM usage guide
   - File descriptions

2. **`rag_systems/README.md`**
   - RAG comparison table
   - Quick start for each system

3. **`rag_systems/rag_pro/README.md`** ⭐
   - Detailed usage guide
   - Exact file paths (absolute & relative)
   - Troubleshooting
   - Configuration
   - Performance metrics

4. **`docs/README.md`**
   - Documentation index
   - Quick links to guides/fixes

---

## ✅ Tested

```bash
cd rag_systems/rag_pro
uv run rag_query.py
```

**Output:**
```
🔄 Loading LLM (GPU priority)...
Loading model...
Model loaded!
   ✅ Llama 3.1 8B loaded

❌ Lỗi: Index not found
💡 Hãy chạy rag_index.py trước để tạo index!
```

**Result:** Import works! ✅ (Index not found is expected - need to run rag_index.py first)

---

## 🚀 Next Steps

1. Run indexing:
```bash
cd rag_systems/rag_pro
uv run rag_index.py --force
```

2. Query:
```bash
uv run rag_query.py
```

---

**Import paths fixed!** 🎉
