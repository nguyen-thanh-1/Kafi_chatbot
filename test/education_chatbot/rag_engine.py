"""
RAG Engine - Complete RAG System with Hybrid Routing
═══════════════════════════════════════════════════════════

ARCHITECTURE:
    User Question
          │
          ▼
    [Question Router]
          │
    ┌─────┴─────────────┐
    │                   │
    ▼                   ▼
  rag_lite           rag_pro
  (fast 3-5s)        (deep 10-20s)
    │                   │
    ▼                   ▼
  LLM + Prior      Strict RAG
  Knowledge        (No hallucination)

═══════════════════════════════════════════════════════════
"""

import os
import sys
import time
import json
import hashlib
import pickle
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
STORAGE_DIR = "./storage"
UPLOADS_DIR = "./uploads"
DATA_DIR = "./data"

INDEX_FILE = os.path.join(STORAGE_DIR, "faiss_index.pkl")
CHUNKS_FILE = os.path.join(STORAGE_DIR, "chunks.json")
TRACKER_FILE = os.path.join(STORAGE_DIR, "files.json")
EMBEDDING_CACHE_FILE = os.path.join(STORAGE_DIR, "embedding_cache.pkl")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Recursive Chunking Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Model Config
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval Config
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 3

# Context Config
MAX_CONTEXT_TOKENS = 1200
TOKENS_PER_CHAR = 0.25

# LLM Config
LLM_MAX_TOKENS = 700
LLM_TEMPERATURE = 0.21

# Routing Config
SIMILARITY_THRESHOLD = 0.5

# Keywords for routing
RAG_PRO_KEYWORDS = [
    "theo tài liệu", "trong sách", "trong tài liệu", "theo sách",
    "chương", "trang", "section", "chapter", "page",
    "được định nghĩa", "được mô tả", "được giải thích",
    "so sánh trong tài liệu", "trích dẫn", "quote",
    "theo như", "dựa theo", "như đã nói"
]


# ═══════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════
HYBRID_PROMPT = """Based on the following context, answer the question.

RULES:
1. Prefer using the provided context if relevant
2. If context is insufficient, you may use general AI knowledge
3. Clearly indicate when the answer is based on general knowledge
4. Answer in the same language as the question

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

STRICT_PROMPT = """Based on the following context, answer the question accurately.

IMPORTANT RULES:
1. ONLY use information from the context below
2. If the answer is NOT in the context, say "Tôi không tìm thấy thông tin này trong tài liệu."
3. Be specific and cite which part of the context you're using
4. Answer in the same language as the question

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

NO_CONTEXT_PROMPT = """Answer the following question using your general AI knowledge.

RULES:
1. Be accurate and educational
2. Answer in the same language as the question
3. If you're unsure, indicate your uncertainty

QUESTION: {question}

ANSWER:"""


# ═══════════════════════════════════════════════════════════
# FILE PROCESSOR
# ═══════════════════════════════════════════════════════════
def read_pdf(file_path: str) -> str:
    import pdfplumber
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def read_text(file_path: str) -> str:
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')


def read_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    return read_text(file_path)


def chunk_text_recursive(text: str, chunk_size: int = CHUNK_SIZE, 
                         chunk_overlap: int = CHUNK_OVERLAP,
                         separators: List[str] = None) -> List[str]:
    """Recursive Character Text Splitter"""
    if separators is None:
        separators = SEPARATORS.copy()
    
    if not text:
        return []
    
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    for i, separator in enumerate(separators):
        if separator == "":
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - chunk_overlap
            return chunks
        
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                if len(part) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    sub_chunks = chunk_text_recursive(part, chunk_size, chunk_overlap, separators[i+1:])
                    chunks.extend(sub_chunks)
                elif len(current_chunk) + len(separator) + len(part) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
                else:
                    if current_chunk:
                        current_chunk += separator + part
                    else:
                        current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            if chunks:
                return chunks
    
    return [text.strip()] if text.strip() else []


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# ═══════════════════════════════════════════════════════════
# MODELS (Lazy Loading)
# ═══════════════════════════════════════════════════════════
_embedder = None
_reranker = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch
        print("📥 Loading Embedding model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"✅ Embedding ready ({device.upper()})")
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from flashrank import Ranker
        print("📥 Loading FlashRank...")
        _reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=STORAGE_DIR)
        print("✅ FlashRank ready (ONNX)")
    return _reranker


def get_llm():
    global _llm
    if _llm is None:
        print("📥 Loading Llama 3.1 8B...")
        from llm_engine import EducationalLLM
        _llm = EducationalLLM()
        print("✅ LLM ready (GPU)")
    return _llm


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True)


def embed_query(query: str) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode([query], normalize_embeddings=True)[0]


def rerank(query: str, chunks: List[str], top_k: int = TOP_K_RERANK) -> List[Tuple[str, float]]:
    if not chunks:
        return []
    
    from flashrank import RerankRequest
    ranker = get_reranker()
    passages = [{"text": chunk} for chunk in chunks]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    return [(r["text"], r["score"]) for r in results[:top_k]]


# ═══════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════
class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.files = {}
        
        os.makedirs(STORAGE_DIR, exist_ok=True)
        self.load()
    
    def load(self) -> bool:
        try:
            if os.path.exists(INDEX_FILE):
                with open(INDEX_FILE, 'rb') as f:
                    self.index = pickle.load(f)
            
            if os.path.exists(CHUNKS_FILE):
                with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = data.get('chunks', [])
                    self.chunk_metadata = data.get('metadata', [])
            
            if os.path.exists(TRACKER_FILE):
                with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
                    self.files = json.load(f)
            
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def save(self):
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump(self.index, f)
        
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'chunks': self.chunks, 'metadata': self.chunk_metadata}, 
                     f, ensure_ascii=False, indent=2)
        
        with open(TRACKER_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.files, f, ensure_ascii=False, indent=2)
    
    def add_file(self, filepath: str, filename: str) -> int:
        import faiss
        
        text = read_file(filepath)
        chunks = chunk_text_recursive(text)
        
        if not chunks:
            return 0
        
        embeddings = embed_texts(chunks)
        dim = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings.astype('float32'))
        
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_metadata.append({'source': filename})
        
        self.files[filename] = {
            'hash': get_file_hash(filepath),
            'chunks': len(chunks),
            'indexed_at': datetime.now().isoformat()
        }
        
        self.save()
        return len(chunks)
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[str]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = embed_query(query).reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results
    
    def get_stats(self) -> dict:
        return {
            'files': len(self.files),
            'chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0
        }


# ═══════════════════════════════════════════════════════════
# QUESTION ROUTER
# ═══════════════════════════════════════════════════════════
class QuestionRouter:
    def classify(self, question: str, context_score: float = 0.0) -> str:
        question_lower = question.lower()
        
        for keyword in RAG_PRO_KEYWORDS:
            if keyword in question_lower:
                return "rag_pro"
        
        if context_score < SIMILARITY_THRESHOLD:
            return "llm_only"
        
        return "rag_lite"
    
    def get_prompt(self, mode: str, question: str, context: str) -> str:
        if mode == "rag_pro":
            return STRICT_PROMPT.format(context=context, question=question)
        elif mode == "rag_lite":
            return HYBRID_PROMPT.format(context=context, question=question)
        else:
            return NO_CONTEXT_PROMPT.format(question=question)


# ═══════════════════════════════════════════════════════════
# RAG HYBRID ENGINE
# ═══════════════════════════════════════════════════════════
class RAGHybrid:
    def __init__(self):
        self.vector_store = VectorStore()
        self.router = QuestionRouter()
    
    def preload_lite(self):
        """Preload models for fast queries"""
        get_embedder()
        get_reranker()
        get_llm()
    
    def index_file(self, filepath: str, filename: str) -> int:
        return self.vector_store.add_file(filepath, filename)
    
    def get_stats(self) -> dict:
        return self.vector_store.get_stats()
    
    def query(self, question: str, verbose: bool = False) -> str:
        """Query and return answer"""
        answer, _ = self.query_with_mode(question, verbose)
        return answer
    
    def query_with_mode(self, question: str, verbose: bool = False) -> Tuple[str, str]:
        """Query and return (answer, mode)"""
        import torch
        
        # Get initial classification
        initial_mode = self.router.classify(question)
        
        # Get context
        retrieved = self.vector_store.search(question, TOP_K_RETRIEVE)
        
        if not retrieved:
            # No documents - use LLM only
            mode = "llm_only"
            prompt = self.router.get_prompt(mode, question, "")
        else:
            # Rerank
            reranked = rerank(question, retrieved, TOP_K_RERANK)
            
            if not reranked:
                mode = "llm_only"
                prompt = self.router.get_prompt(mode, question, "")
            else:
                # Get top score for routing decision
                top_score = reranked[0][1]
                
                # Re-evaluate mode based on score
                if initial_mode == "rag_pro":
                    mode = "rag_pro"
                elif top_score < SIMILARITY_THRESHOLD:
                    mode = "llm_only"
                else:
                    mode = "rag_lite"
                
                # Build context
                context_parts = []
                for i, (chunk, score) in enumerate(reranked, 1):
                    context_parts.append(f"[Đoạn {i}]\n{chunk}")
                context = "\n\n---\n\n".join(context_parts)
                
                prompt = self.router.get_prompt(mode, question, context)
        
        # Generate answer
        llm = get_llm()
        
        system_prompt = "You are a helpful educational AI assistant. Be accurate, clear, and helpful."
        
        answer = llm.generate(prompt, system_prompt=system_prompt,
                            max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        return answer, mode
