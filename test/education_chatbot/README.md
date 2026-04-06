# Education Chatbot - Complete RAG System

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
(fast 3-5s)       (deep 10-20s)
 │                   │
 ▼                   ▼
LLM + Prior      Strict RAG
Knowledge        (No hallucination)
```

---

## 🚀 Quick Start

```bash
cd education_chatbot
streamlit run app.py
```

---

## 📁 File Structure

```
education_chatbot/
├── app.py           # Streamlit UI (main)
├── rag_engine.py    # RAG Hybrid với Question Router
├── llm_engine.py    # Llama 3.1 8B wrapper
├── prompts.py       # Subject detection & prompts
├── utils.py         # Language routing & utilities
├── storage/         # FAISS index & chunks
├── uploads/         # Uploaded files
├── data/            # Pre-indexed data
└── README.md
```

---

## 📊 Components

| Component | Model | Notes |
|-----------|-------|-------|
| **Embedding** | all-MiniLM-L6-v2 | GPU, 384 dim |
| **Reranker** | FlashRank | ONNX (CPU) |
| **LLM** | Llama 3.1 8B | 4-bit quantized |
| **Chunking** | Recursive | 1000 chars |

---

## 🔀 Routing Rules

| Question Type | Mode | Prompt |
|---------------|------|--------|
| "NLP là gì?" | ⚡ Hybrid | Context + LLM knowledge |
| "Theo tài liệu, NLP là gì?" | 📚 Strict | Only document |
| Low similarity | 🤖 LLM Only | General AI knowledge |

---

## 📚 Features

- ✅ Upload PDF, TXT, MD, CSV
- ✅ Auto-indexing with progress
- ✅ Subject detection (Math, Physics, Chemistry, English)
- ✅ Language detection (Vietnamese/English)
- ✅ 2-Stage RAG with intelligent routing
- ✅ Mode indicator in chat
- ✅ Dark theme UI

---

## ⚙️ Configuration

In `rag_engine.py`:

```python
# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 3

# Routing
SIMILARITY_THRESHOLD = 0.5

# LLM
LLM_MAX_TOKENS = 700
LLM_TEMPERATURE = 0.21
```

---

## 📝 Prompts

### Hybrid Mode
```
RULES:
1. Prefer using the provided context if relevant
2. If context is insufficient, you may use general AI knowledge
3. Clearly indicate when the answer is based on general knowledge
```

### Strict Mode
```
RULES:
1. ONLY use information from the context below
2. If the answer is NOT in the context, say "Tôi không tìm thấy..."
```

---

## 🎓 Subject Detection

Tự động nhận diện môn học:
- 🔢 Math: "phương trình", "tính", "đạo hàm"...
- ⚛️ Physics: "lực", "chuyển động", "điện"...
- 🧪 Chemistry: "phản ứng", "mol", "axit"...
- 🔤 English: "grammar", "tense", "vocabulary"...
