# RAG Hybrid - 2-Stage Question Routing

## 🎯 Architecture

```
User Question
      │
      ▼
[Question Router]
      │
 ┌────┴──────────────┐
 │                   │
 ▼                   ▼
rag_lite          rag_pro
(fast)            (deep)
 │                   │
 ▼                   ▼
LLM + Prior      Strict RAG
Knowledge        (No hallucination)
```

---

## 📊 Routing Rules

| Question Type | Mode | Prompt |
|---------------|------|--------|
| "NLP là gì?" | rag_lite | Hybrid (context + LLM knowledge) |
| "Theo tài liệu, NLP gồm những bước nào?" | rag_pro | Strict (only document) |
| Low similarity score | llm_only | LLM general knowledge |

---

## 🔑 Keywords

### → rag_pro (strict):
- "theo tài liệu", "trong sách"
- "chương", "trang", "section"
- "được định nghĩa", "trích dẫn"

### → rag_lite (hybrid):
- "là gì", "định nghĩa"
- "giải thích", "tại sao"
- "ví dụ", "ứng dụng"

---

## 🚀 Quick Start

```bash
cd rag_systems/rag_hybrid
uv run rag_query.py
```

---

## 📝 Prompts

### Hybrid (rag_lite):
```
RULES:
1. Prefer using the provided context if relevant
2. If context is insufficient, you may use general AI knowledge
3. Clearly indicate when the answer is based on general knowledge
```

### Strict (rag_pro):
```
RULES:
1. ONLY use information from the context below
2. If the answer is NOT in the context, say "Tôi không tìm thấy..."
3. Be specific and cite which part of the context
```

---

## ⚡ Performance

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| rag_lite | 3-5s | ⭐⭐⭐ | General Q&A |
| rag_pro | 10-20s | ⭐⭐⭐⭐⭐ | Document-specific |
| llm_only | 2-3s | ⭐⭐ | No context needed |

---

## 📁 Files

```
rag_hybrid/
├── rag_hybrid.py   # Main implementation
├── rag_query.py    # Query script
└── README.md       # This file
```

---

## 🔧 Configuration

In `rag_hybrid.py`:

```python
SIMILARITY_THRESHOLD = 0.5  # Below → use LLM only

RAG_PRO_KEYWORDS = [
    "theo tài liệu", "trong sách", ...
]
```
