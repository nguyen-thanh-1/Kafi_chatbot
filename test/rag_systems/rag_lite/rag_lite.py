"""
RAG Lite - Lightweight RAG System
═══════════════════════════════════════════════════════════

COMPONENTS:
- Embedding: all-MiniLM-L6-v2 (fast, lightweight, 384 dim)
- Reranker: FlashRank (ONNX-based, very fast)
- Chunking: Recursive Character Splitting

PERFORMANCE:
- Embedding: ~5x faster than BGE-M3
- Reranker: ~10x faster than CrossEncoder
- VRAM: ~2GB total (very light)

USE CASE:
- Quick prototyping
- Low resource environments
- Real-time applications

═══════════════════════════════════════════════════════════
"""

import os
import json
import hashlib
import time
import pickle
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
COURSES_FOLDER = os.path.join(PROJECT_ROOT, "data", "courses_v2")
RAG_STORAGE = os.path.join(PROJECT_ROOT, "storage", "rag_storage_lite")
INDEX_FILE = os.path.join(RAG_STORAGE, "faiss_index.pkl")
CHUNKS_FILE = os.path.join(RAG_STORAGE, "chunks.json")
TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")
EMBEDDING_CACHE_FILE = os.path.join(RAG_STORAGE, "embedding_cache.pkl")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# ═══════════════════════════════════════════════════════════
# RECURSIVE CHUNKING CONFIG
# ═══════════════════════════════════════════════════════════
CHUNK_SIZE = 1000        # characters (not words)
CHUNK_OVERLAP = 200      # characters overlap
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Recursive separators

# ═══════════════════════════════════════════════════════════
# MODEL CONFIG
# ═══════════════════════════════════════════════════════════
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dim, very fast
EMBEDDING_DIM = 384

# Retrieval config
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 3

# Context config
MAX_CONTEXT_TOKENS = 1200
TOKENS_PER_CHAR = 0.25  # ~4 chars per token

# LLM config
LLM_MAX_TOKENS = 700
LLM_TEMPERATURE = 0.21


# ═══════════════════════════════════════════════════════════
# RECURSIVE TEXT CHUNKER
# ═══════════════════════════════════════════════════════════
def chunk_text_recursive(
    text: str, 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = None
) -> List[str]:
    """
    Recursive Character Text Splitter
    
    Strategy:
    1. Try to split by largest separator first (\n\n)
    2. If chunks too large, try next separator (\n)
    3. Continue until chunks fit or use character split
    
    This preserves document structure while ensuring consistent chunk sizes.
    """
    if separators is None:
        separators = SEPARATORS.copy()
    
    if not text:
        return []
    
    # Base case: text fits in chunk
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    # Try each separator
    for i, separator in enumerate(separators):
        if separator == "":
            # Last resort: character split
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
            # Split by this separator
            parts = text.split(separator)
            
            # Merge small parts, split large parts
            chunks = []
            current_chunk = ""
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # If part alone is too large, recursively split it
                if len(part) > chunk_size:
                    # Flush current chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # Recursively split with remaining separators
                    sub_chunks = chunk_text_recursive(
                        part, 
                        chunk_size, 
                        chunk_overlap, 
                        separators[i+1:]
                    )
                    chunks.extend(sub_chunks)
                
                # If adding part exceeds chunk size
                elif len(current_chunk) + len(separator) + len(part) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
                
                # Add part to current chunk
                else:
                    if current_chunk:
                        current_chunk += separator + part
                    else:
                        current_chunk = part
            
            # Don't forget the last chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            if chunks:
                return chunks
    
    return [text.strip()] if text.strip() else []


# ═══════════════════════════════════════════════════════════
# FILE READERS
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


# ═══════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════
class EmbeddingCache:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load()
        self.hits = 0
        self.misses = 0
    
    def _load(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"   📦 Loaded cache: {len(cache)} embeddings")
                    return cache
            except Exception as e:
                print(f"   ⚠️ Cache load failed: {e}")
                return {}
        return {}
    
    def save(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        print(f"   💾 Cache saved: {len(self.cache)} embeddings")
    
    def get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        hash_key = self.get_hash(text)
        emb = self.cache.get(hash_key)
        if emb is not None:
            self.hits += 1
        else:
            self.misses += 1
        return emb
    
    def set(self, text: str, embedding: np.ndarray):
        hash_key = self.get_hash(text)
        self.cache[hash_key] = embedding


# ═══════════════════════════════════════════════════════════
# INDEX TRACKER
# ═══════════════════════════════════════════════════════════
class IndexTracker:
    def __init__(self, tracker_file: str):
        self.tracker_file = tracker_file
        self.indexed_files = self._load()
    
    def _load(self) -> dict:
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save(self):
        os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.indexed_files, f, indent=2, ensure_ascii=False)
    
    def _get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def needs_indexing(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        filename = os.path.basename(file_path)
        current_hash = self._get_file_hash(file_path)
        if filename not in self.indexed_files:
            return True
        if self.indexed_files[filename].get('hash') != current_hash:
            return True
        return False
    
    def mark_indexed(self, file_path: str, chunk_count: int):
        filename = os.path.basename(file_path)
        self.indexed_files[filename] = {
            'hash': self._get_file_hash(file_path),
            'indexed_at': datetime.now().isoformat(),
            'size_bytes': os.path.getsize(file_path),
            'chunks': chunk_count,
        }
        self._save()
    
    def get_indexed_count(self) -> int:
        return len(self.indexed_files)
    
    def get_total_chunks(self) -> int:
        return sum(f.get('chunks', 0) for f in self.indexed_files.values())


# ═══════════════════════════════════════════════════════════
# EMBEDDING MODEL (MiniLM-L6-v2) - LIGHTWEIGHT
# ═══════════════════════════════════════════════════════════
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch
        print(f"   📥 Loading {EMBEDDING_MODEL}...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"   ✅ Embedding model loaded ({device.upper()})")
    return _embedder


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


def embed_texts_cached(texts: List[str], cache: EmbeddingCache, batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    
    embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    to_embed = []
    to_embed_indices = []
    
    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            embeddings[i] = cached
        else:
            to_embed.append(text)
            to_embed_indices.append(i)
    
    if to_embed:
        print(f"   🔄 Embedding {len(to_embed)} new chunks (cached: {len(texts) - len(to_embed)})...")
        new_embeddings = embed_texts(to_embed, batch_size=batch_size)
        
        for i, (text, emb) in enumerate(zip(to_embed, new_embeddings)):
            idx = to_embed_indices[i]
            embeddings[idx] = emb
            cache.set(text, emb)
    else:
        print(f"   ✅ All {len(texts)} chunks from cache!")
    
    return embeddings


def embed_query(query: str) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode([query], normalize_embeddings=True)[0]


# ═══════════════════════════════════════════════════════════
# RERANKER (FlashRank) - VERY FAST
# ═══════════════════════════════════════════════════════════
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from flashrank import Ranker, RerankRequest
        print(f"   📥 Loading FlashRank...")
        
        _reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=RAG_STORAGE)
        print(f"   ✅ FlashRank loaded (ONNX)")
    return _reranker


def rerank(query: str, chunks: List[str], top_k: int = TOP_K_RERANK) -> List[Tuple[str, float]]:
    """Rerank chunks using FlashRank"""
    if not chunks:
        return []
    
    from flashrank import RerankRequest
    
    ranker = get_reranker()
    
    # Prepare passages
    passages = [{"text": chunk} for chunk in chunks]
    
    # Create rerank request
    request = RerankRequest(query=query, passages=passages)
    
    # Rerank
    results = ranker.rerank(request)
    
    # Get top_k
    scored_chunks = [(r["text"], r["score"]) for r in results[:top_k]]
    
    return scored_chunks


# ═══════════════════════════════════════════════════════════
# VECTOR STORE (FAISS)
# ═══════════════════════════════════════════════════════════
class VectorStore:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        os.makedirs(storage_dir, exist_ok=True)
    
    def load(self) -> bool:
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            try:
                import faiss
                with open(INDEX_FILE, 'rb') as f:
                    self.index = pickle.load(f)
                with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = data.get('chunks', [])
                    self.chunk_metadata = data.get('metadata', [])
                print(f"   📦 Loaded index: {len(self.chunks)} chunks")
                return True
            except Exception as e:
                print(f"   ⚠️ Load failed: {e}")
        return False
    
    def save(self):
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump(self.index, f)
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }, f, ensure_ascii=False, indent=2)
    
    def add_chunks(self, chunks: List[str], source_file: str, embeddings: np.ndarray):
        import faiss
        
        if not chunks:
            return
        
        dim = embeddings.shape[1]
        
        if self.index is None:
            # Use simple Flat IP index (MiniLM is small)
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings.astype('float32'))
        
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_metadata.append({'source': source_file})
        
        self.save()
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[str]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = embed_query(query).reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results
    
    def clear(self):
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)


# ═══════════════════════════════════════════════════════════
# LLM (LLAMA 3.1 8B)
# ═══════════════════════════════════════════════════════════
_llm = None

def get_llm():
    """Load LLM immediately (preload at startup)"""
    global _llm
    if _llm is None:
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print(f"   📥 Loading Llama 3.1 8B...")
        from llm_models.Llama_3_1_8B_Instruct_v2 import generate_response, _load_model
        
        _load_model()
        
        _llm = generate_response
        print(f"   ✅ Llama 3.1 8B loaded (GPU)")
    return _llm


def truncate_context(chunks: List[Tuple[str, float]], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Truncate context to prevent KV cache OOM"""
    context_parts = []
    total_tokens = 0
    max_chunks = min(len(chunks), TOP_K_RERANK)
    
    for i, (chunk, score) in enumerate(chunks[:max_chunks], 1):
        chunk_tokens = int(len(chunk) * TOKENS_PER_CHAR)
        
        if total_tokens + chunk_tokens > max_tokens:
            remaining_tokens = max_tokens - total_tokens
            remaining_chars = int(remaining_tokens / TOKENS_PER_CHAR)
            truncated_chunk = chunk[:remaining_chars] + "..."
            context_parts.append(f"[Đoạn {i}]\n{truncated_chunk}")
            total_tokens = max_tokens
            break
        
        context_parts.append(f"[Đoạn {i}]\n{chunk}")
        total_tokens += chunk_tokens
    
    print(f"   ✅ Context: {len(context_parts)} chunks, {total_tokens} tokens")
    
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, context_chunks: List[Tuple[str, float]]) -> str:
    """Generate answer from context chunks"""
    llm = get_llm()
    
    if not context_chunks:
        return "Tôi không tìm thấy thông tin liên quan trong tài liệu."
    
    context = truncate_context(context_chunks, MAX_CONTEXT_TOKENS)
    
    prompt = f"""Based on the following context, answer the question accurately.

IMPORTANT RULES:
1. ONLY use information from the context below
2. If the answer is NOT in the context, say "Tôi không tìm thấy thông tin này trong tài liệu."
3. Be specific and cite which part of the context you're using
4. Answer in the same language as the question

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    system_prompt = "You are a helpful assistant that answers questions based ONLY on the provided context. Never make up information."
    
    response = llm(
        user_input=prompt,
        history=[],
        system_prompt=system_prompt,
        max_new_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    
    return response


# ═══════════════════════════════════════════════════════════
# RAG LITE PIPELINE
# ═══════════════════════════════════════════════════════════
class RAGLite:
    def __init__(self):
        self.vector_store = VectorStore(RAG_STORAGE)
        self.tracker = IndexTracker(TRACKER_FILE)
        self.cache = EmbeddingCache(EMBEDDING_CACHE_FILE)
    
    def index_file(self, file_path: str) -> int:
        """Index a single file"""
        text = read_file(file_path)
        
        # Recursive chunking
        chunks = chunk_text_recursive(text)
        
        if chunks:
            embeddings = embed_texts_cached(chunks, self.cache)
            self.vector_store.add_chunks(chunks, os.path.basename(file_path), embeddings)
            self.tracker.mark_indexed(file_path, len(chunks))
        
        return len(chunks)
    
    def index_folder(self, folder: str, force: bool = False) -> int:
        """Index all files in folder"""
        if force:
            self.vector_store.clear()
            self.tracker.indexed_files = {}
            self.tracker._save()
        
        if not os.path.exists(folder):
            print(f"⚠️ Folder không tồn tại: {folder}")
            return 0
        
        all_files = [f for f in os.listdir(folder) 
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        
        new_files = [f for f in all_files 
                     if self.tracker.needs_indexing(os.path.join(folder, f))]
        
        if not new_files:
            print(f"✅ Không có file mới. Database: {self.tracker.get_indexed_count()} files, {self.tracker.get_total_chunks()} chunks")
            return 0
        
        print(f"\n🆕 Phát hiện {len(new_files)} file cần index:")
        
        total_chunks = 0
        for i, filename in enumerate(new_files, 1):
            file_path = os.path.join(folder, filename)
            try:
                print(f"   [{i}/{len(new_files)}] {filename}...", end=" ", flush=True)
                start = time.time()
                chunks = self.index_file(file_path)
                elapsed = time.time() - start
                print(f"✅ {chunks} chunks ({elapsed:.1f}s)")
                total_chunks += chunks
            except Exception as e:
                print(f"❌ {e}")
        
        self.cache.save()
        return total_chunks
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Query with VRAM-safe execution"""
        import torch
        start = time.time()
        
        if verbose:
            print(f"   🔍 Searching...")
        retrieved_chunks = self.vector_store.search(question, TOP_K_RETRIEVE)
        
        if not retrieved_chunks:
            return "Tôi không tìm thấy thông tin liên quan trong database."
        
        if verbose:
            print(f"   📄 Found {len(retrieved_chunks)} chunks")
        
        if verbose:
            print(f"   🎯 Reranking with FlashRank...")
        reranked = rerank(question, retrieved_chunks, TOP_K_RERANK)
        
        if verbose:
            print(f"   ✅ Selected {len(reranked)} best chunks")
        
        if verbose:
            print(f"   🤖 Generating answer...")
        answer = generate_answer(question, reranked)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        elapsed = time.time() - start
        if verbose:
            print(f"   ⏱️ Total: {elapsed:.1f}s")
        
        return answer
    
    def load(self) -> bool:
        return self.vector_store.load()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Lite - Lightweight RAG System")
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index')
    parser.add_argument('--query', '-q', type=str, help='Single query mode')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG LITE - Lightweight RAG System")
    print("═" * 60)
    print(f"   📊 Embedding: {EMBEDDING_MODEL}")
    print(f"   🎯 Reranker:  FlashRank (ONNX)")
    print(f"   🤖 LLM:       Llama 3.1 8B")
    print(f"   ⚡ Chunking:  Recursive ({CHUNK_SIZE} chars)")
    print("═" * 60)
    
    print("\n🔄 Loading models...")
    rag = RAGLite()
    
    if not args.force:
        rag.load()
    
    get_embedder()
    get_reranker()
    get_llm()
    
    print("\n" + "═" * 60)
    print("📁 INDEXING")
    print("═" * 60)
    rag.index_folder(COURSES_FOLDER, force=args.force)
    
    if args.query:
        print("\n" + "═" * 60)
        print("🔍 QUERY")
        print("═" * 60)
        print(f"\n❓ {args.query}")
        answer = rag.query(args.query)
        print(f"\n🤖 Answer:\n{answer}")
        return
    
    print("\n" + "═" * 60)
    print("💬 INTERACTIVE MODE")
    print("═" * 60)
    print("Gõ câu hỏi. 'exit' để thoát.")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🧑 Bạn: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break
            
            if not question:
                continue
            
            print("\n🤖 Đang xử lý...")
            answer = rag.query(question)
            print(f"\n📝 Trả lời:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break


if __name__ == "__main__":
    main()
