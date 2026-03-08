import os
import sys
import json
import faiss
import fitz
import torch
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

sys.path.append(r"C:\Users\Admin\Desktop\Special_Subject_AI\llm_models")
from Qwen3_VL_8B_Instruct import QwenVLChat

# Initialize QwenVLChat globally (or in a lazy-load pattern to save memory if needed)
# However, we'll initialize it here directly like the existing models.
qwen_vl = None

def init_qwen():
    global qwen_vl
    if qwen_vl is None:
        qwen_vl = QwenVLChat()

# =========================
# CONFIG
# =========================

PDF_PATH = r"C:\Users\Admin\Desktop\Special_Subject_AI\data\courses_v2\nlp-book.pdf"
INDEX_PATH = "faiss.index"
META_PATH = "chunks.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

CHUNK_SIZE = 1000
OVERLAP = 150
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

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

def chunk_text(text):
    tokens = tokenizer.encode(text)
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

embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

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

reranker = CrossEncoder(RERANKER_MODEL, device=device)

def rerank(query, candidates):
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores),
                    key=lambda x: x[1],
                    reverse=True)
    return [x[0] for x in ranked[:TOP_K_RERANK]]

# =========================
# LLM (Qwen3 VL)
# =========================

def generate_answer(query, contexts):
    init_qwen()
    context_text = "\n\n".join(contexts)

    user_input = f"Dựa vào thông tin ngữ cảnh sau:\n{context_text}\n\nHãy trả lời câu hỏi một cách chính xác: {query}"
    
    # Reset chat to avoid context leaking between requests
    qwen_vl.reset_chat()
    
    # Run the chat generation (this function is synchronous because it waits for the loop)
    # We intercept STDOUT natively in the class, but we can just get history later
    qwen_vl.chat_stream(user_input)
    
    # Retrieve the assistant's latest response from history
    latest_msg = qwen_vl.history[-1]
    if latest_msg["role"] == "assistant":
        return latest_msg["content"][0]["text"].strip()
    
    return ""

# =========================
# CACHE
# =========================

query_cache = {}

# =========================
# BUILD OR LOAD SYSTEM
# =========================

if os.path.exists(INDEX_PATH):
    print("Loading existing index...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
else:
    print("Building new index...")
    if os.path.exists(PDF_PATH):
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)

        index = build_index(embeddings)

        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f)
    else:
        print(f"File {PDF_PATH} does not exist.")
        chunks = []
        index = None

# =========================
# RETRIEVAL
# =========================

def retrieve(query):
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

    top_contexts = rerank(query, candidates)

    query_cache[query] = top_contexts
    return top_contexts

# =========================
# FASTAPI SERVER
# =========================

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    contexts = retrieve(req.question)
    answer = generate_answer(req.question, contexts)
    return {"answer": answer}
