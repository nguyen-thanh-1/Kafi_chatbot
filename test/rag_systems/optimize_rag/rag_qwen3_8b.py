import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import json
import logging
import faiss
import fitz
import torch
import numpy as np
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append(r"C:\Users\Admin\Desktop\Special_Subject_AI\llm_models")
from Qwen3_8B import QwenChat

# =========================
# CONFIG
# =========================

PDF_PATH = r"C:\Users\Admin\Desktop\Special_Subject_AI\data\courses_v2\nlp-book.pdf"
INDEX_PATH = "faiss_qwen3_8b.index"
META_PATH = "chunks_qwen3_8b.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 400
OVERLAP = 50
BATCH_SIZE = 64
TOP_K_RETRIEVE = 8
TOP_K_RERANK = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# PDF EXTRACTION
# =========================

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# =========================
# CHUNKING
# =========================

def chunk_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        start += CHUNK_SIZE - OVERLAP

    return chunks

# =========================
# EMBEDDING + INDEX
# =========================

def embed_chunks(chunks):
    return embed_model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def build_index(embeddings):
    dim = embeddings.shape[1]
    nlist = min(100, max(1, len(embeddings) // 39)) if len(embeddings) > 0 else 1

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = min(10, nlist)
    return index

# =========================
# RERANKER
# =========================

def rerank(query, candidates):
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked[:TOP_K_RERANK]]

# =========================
# LLM (Qwen3-8B)
# =========================

def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    # Mirror LLaMA RAG: strict system_prompt + clear Context/Question/Answer format
    qwen.system_prompt = (
        "You are an expert academic assistant. "
        "Use ONLY the provided context to answer the question. "
        "If the context does not contain relevant information, "
        "explicitly say you cannot find it in the provided documents. "
        "Do NOT use any external knowledge outside the context."
    )
    qwen.reset_chat()

    # Same format as rag_llama.py: Context / Question / Answer
    user_input = f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"

    # Temporarily lower temperature to 0.2 (same as LLaMA) to reduce hallucination
    original_do_sample = True
    qwen.chat_stream.__func__  # check it's a regular method (no override needed)

    # Patch generation to use low temperature like LLaMA (temperature=0.2, do_sample=False)
    import types
    original_chat_stream = qwen.chat_stream

    def low_temp_chat_stream(user_input_text):
        qwen.history.append({"role": "user", "content": user_input_text})
        text = qwen.tokenizer.apply_chat_template(
            qwen.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = qwen.tokenizer([text], return_tensors="pt").to(qwen.model.device)
        from transformers import TextIteratorStreamer
        from threading import Thread
        streamer = TextIteratorStreamer(qwen.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=False,      # greedy, same as LLaMA
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
            pad_token_id=qwen.tokenizer.pad_token_id or qwen.tokenizer.eos_token_id,
            eos_token_id=qwen.tokenizer.eos_token_id
        )
        thread = Thread(target=qwen.model.generate, kwargs=generation_kwargs)
        thread.start()
        full_response = ""
        for new_text in streamer:
            full_response += new_text
        thread.join()
        qwen.history[-1]["content"] = user_input_text
        qwen.history.append({"role": "assistant", "content": full_response})

    low_temp_chat_stream(user_input)

    latest_msg = qwen.history[-1]
    if latest_msg["role"] == "assistant":
        return latest_msg["content"].strip()

    return ""

# =========================
# CACHE
# =========================

query_cache = {}

# =========================
# GLOBALS & LIFESPAN
# =========================

index = None
chunks = []
tokenizer = None
embed_model = None
reranker = None
qwen = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, chunks, tokenizer, embed_model, reranker, qwen

    print("Loading models (Embedding, Reranker, Qwen3-8B)...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)

    # Tải Qwen3-8B với Non-Thinking mode (nhanh hơn, phù hợp RAG)
    # Đặt enable_thinking=True nếu muốn bật chế độ suy luận sâu
    qwen = QwenChat(enable_thinking=False)

    if os.path.exists(INDEX_PATH):
        print("Loading existing index...")
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        print("No existing index found. Ready for upload.")
    yield

# =========================
# RETRIEVAL
# =========================

def retrieve(query):
    """Truy xuất ngữ cảnh liên quan từ FAISS index và rerank."""
    if not index or not chunks:
        return []

    if query in query_cache:
        return query_cache[query]

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    D, I = index.search(query_embedding, TOP_K_RETRIEVE)
    candidates = [chunks[i] for i in I[0] if i < len(chunks)]

    contexts = rerank(query, candidates)
    query_cache[query] = contexts
    return contexts

# =========================
# FASTAPI SERVER
# =========================

app = FastAPI(
    title="RAG API - Qwen3-8B",
    description="Retrieval-Augmented Generation API sử dụng Qwen3-8B làm LLM backbone.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """Upload PDF và xây dựng FAISS index mới."""
    global index, chunks
    try:
        with open(PDF_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("Building new index...")
        text = extract_text_from_pdf(PDF_PATH)
        new_chunks = chunk_text(text)
        embeddings = embed_chunks(new_chunks)

        new_index = build_index(embeddings)

        faiss.write_index(new_index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(new_chunks, f)

        chunks = new_chunks
        index = new_index

        # Xóa cache cũ khi có tài liệu mới
        query_cache.clear()

        return {
            "message": f"File '{file.filename}' uploaded and processed successfully. Index updated.",
            "chunks_count": len(chunks)
        }
    except Exception as e:
        return {"error": str(e)}

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    """Trả lời câu hỏi dựa trên tài liệu đã upload."""
    contexts = retrieve(req.question)
    answer = generate_answer(req.question, contexts)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
