import os
import sys
import json
import faiss
import fitz
import torch
import numpy as np
import shutil
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

sys.path.append(r"C:\Users\Admin\Desktop\Special_Subject_AI\llm_models")
from Llama_3_1_8B_Instruct_v2 import generate_response, _load_model

# =========================
# CONFIG
# =========================

PDF_PATH = r"C:\Users\Admin\Desktop\Special_Subject_AI\data\courses_v2\nlp-book.pdf"
INDEX_PATH = "faiss.index"
META_PATH = "chunks.json"
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
# LLM (LLaMA 3.1)
# =========================

def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    system_prompt = "You are an expert academic assistant. Use ONLY the provided context to answer the question."
    user_input = f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"
    
    response = generate_response(
        user_input=user_input,
        system_prompt=system_prompt,
        max_new_tokens=512,
        temperature=0.2
    )

    return response.strip()

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, chunks, tokenizer, embed_model, reranker
    
    print("Loading models (Embedding, Reranker, LLM)...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    reranker = CrossEncoder(RERANKER_MODEL, device=device)
    
    # Force load LLaMA model immediately instead of lazy loading on first request
    _load_model()
    
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

app = FastAPI(lifespan=lifespan)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
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
    contexts = retrieve(req.question)
    answer = generate_answer(req.question, contexts)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
