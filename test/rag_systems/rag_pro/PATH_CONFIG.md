# Path Configuration Guide

## 📂 Current Structure

```
Special_Subject_AI/                    # PROJECT_ROOT
├── rag_systems/
│   └── rag_pro/
│       ├── rag_pro_v2.py             # This file
│       ├── rag_index.py
│       └── rag_query.py
├── data/
│   └── courses/                       # Input documents
└── storage/
    └── rag_storage_pro_v2/           # Index storage
```

---

## ✅ Dynamic Path Calculation

**In `rag_pro_v2.py`:**

```python
# Calculate project root (2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Use relative paths from project root
COURSES_FOLDER = os.path.join(PROJECT_ROOT, "data", "courses")
RAG_STORAGE = os.path.join(PROJECT_ROOT, "storage", "rag_storage_pro_v2")
```

**Path calculation:**
```
__file__                              # rag_systems/rag_pro/rag_pro_v2.py
os.path.dirname(__file__)             # rag_systems/rag_pro
os.path.join(..., '..')               # rag_systems
os.path.join(..., '..', '..')         # Special_Subject_AI (PROJECT_ROOT)
```

---

## 📁 Folder Mapping

| Variable | Path | Purpose |
|----------|------|---------|
| `PROJECT_ROOT` | `C:\Users\Admin\Desktop\Special_Subject_AI` | Project root |
| `COURSES_FOLDER` | `{PROJECT_ROOT}\data\courses` | Input documents |
| `RAG_STORAGE` | `{PROJECT_ROOT}\storage\rag_storage_pro_v2` | Index storage |

---

## 🎯 Benefits

### Before (Hardcoded):
```python
COURSES_FOLDER = "C:\Users\Admin\Desktop\Special_Subject_AI\data\courses_v2"
RAG_STORAGE = "C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\rag_pro\rag_storage_pro_v2"
```

**Problems:**
- ❌ Only works on one machine
- ❌ Breaks if project moves
- ❌ Hard to share with others

### After (Dynamic):
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
COURSES_FOLDER = os.path.join(PROJECT_ROOT, "data", "courses")
RAG_STORAGE = os.path.join(PROJECT_ROOT, "storage", "rag_storage_pro_v2")
```

**Benefits:**
- ✅ Works on any machine
- ✅ Works if project moves
- ✅ Easy to share
- ✅ Can run from any directory

---

## 🚀 Usage

### From any directory:

```bash
# From project root
cd C:\Users\Admin\Desktop\Special_Subject_AI
uv run rag_systems/rag_pro/rag_index.py --force

# From rag_pro folder
cd rag_systems/rag_pro
uv run rag_index.py --force

# From anywhere
cd C:\
uv run C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\rag_pro\rag_index.py --force
```

**All work!** ✅

---

## 📝 Notes

1. **Data folder:** Changed from `courses_v2` to `courses` (standard name)
2. **Storage folder:** Now in `storage/rag_storage_pro_v2/` (organized)
3. **Automatic creation:** Folders created automatically if they don't exist

---

## ✅ Checklist

Before running:
- [ ] Documents in `data/courses/`
- [ ] Run `rag_index.py --force` first time
- [ ] Then run `rag_query.py`

---

**Paths are now portable!** 🎉
