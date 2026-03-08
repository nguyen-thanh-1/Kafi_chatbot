import os
import json
import faiss
import fitz
import torch
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# CONFIG
# =========================

PDF_PATH = r"C:\Users\Admin\Desktop\Special_Subject_AI\data\courses_v2\nlp-book.pdf"
INDEX_PATH = "faiss.index"
META_PATH = "chunks.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

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
    nlist = 100

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 10
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
# LLM
# =========================

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer
)


def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an expert academic assistant.
Use ONLY the provided context.

Context:
{context_text}

Question:
{query}

Answer:
"""

    output = generator(
        prompt,
        max_new_tokens=512,
        do_sample=False
    )

    full_text = output[0]["generated_text"]
    # Chỉ lấy phần answer, bỏ phần prompt
    if "Answer:" in full_text:
        return full_text.split("Answer:")[-1].strip()
    return full_text.strip()


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
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    index = build_index(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

# =========================
# RETRIEVAL
# =========================

def retrieve(query):
    if query in query_cache:
        return query_cache[query]

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    D, I = index.search(query_embedding, TOP_K_RETRIEVE)
    candidates = [chunks[i] for i in I[0]]

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