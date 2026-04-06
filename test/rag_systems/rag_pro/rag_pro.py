"""
RAG Pro - Phiên bản tối ưu nhất
═══════════════════════════════════════════════════════════
Architecture:
  1. Embedding:  bge-m3 (multilingual, best quality)
  2. Vector DB:  FAISS (fast, efficient)
  3. Reranker:   bge-reranker-v2-m3 (accuracy boost)
  4. LLM:        Llama 3.1 8B (local, private)

Chạy: uv run rag_pro.py                    # Index + Query
      uv run rag_pro.py --force            # Re-index
      uv run rag_pro.py --query "câu hỏi"  # Single query
═══════════════════════════════════════════════════════════
"""

import asyncio
import json
import hashlib
import os
import time
import pickle
from datetime import datetime
from typing import List, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
COURSES_FOLDER = "./courses"
RAG_STORAGE = "./rag_storage_pro"
INDEX_FILE = os.path.join(RAG_STORAGE, "faiss_index.pkl")
CHUNKS_FILE = os.path.join(RAG_STORAGE, "chunks.json")
TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Chunking config
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50

# Model config
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Retrieval config
TOP_K_RETRIEVE = 20  # Số chunks lấy từ FAISS
TOP_K_RERANK = 5     # Số chunks sau rerank để đưa vào LLM

# LLM config
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1


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
# TEXT CHUNKER
# ═══════════════════════════════════════════════════════════
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chunk text theo số từ (đơn giản hóa thay vì token)"""
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks


# ═══════════════════════════════════════════════════════════
# EMBEDDING MODEL (BGE-M3)
# ═══════════════════════════════════════════════════════════
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"   📥 Loading {EMBEDDING_MODEL}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"   ✅ Embedding loaded")
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode([query], normalize_embeddings=True)[0]


# ═══════════════════════════════════════════════════════════
# RERANKER (BGE-RERANKER-V2-M3)
# ═══════════════════════════════════════════════════════════
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"   📥 Loading {RERANKER_MODEL}...")
        _reranker = CrossEncoder(RERANKER_MODEL)
        print(f"   ✅ Reranker loaded")
    return _reranker


def rerank(query: str, chunks: List[str], top_k: int = TOP_K_RERANK) -> List[Tuple[str, float]]:
    """Rerank chunks và trả về top_k tốt nhất"""
    if not chunks:
        return []
    
    reranker = get_reranker()
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    
    # Sort by score descending
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_chunks[:top_k]


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
        """Load index từ disk"""
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            try:
                import faiss
                with open(INDEX_FILE, 'rb') as f:
                    self.index = pickle.load(f)
                with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = data.get('chunks', [])
                    self.chunk_metadata = data.get('metadata', [])
                return True
            except Exception as e:
                print(f"   ⚠️ Load failed: {e}")
        return False
    
    def save(self):
        """Save index to disk"""
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump(self.index, f)
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }, f, ensure_ascii=False, indent=2)
    
    def add_chunks(self, chunks: List[str], source_file: str):
        """Thêm chunks vào index"""
        import faiss
        
        if not chunks:
            return
        
        # Embed
        embeddings = embed_texts(chunks)
        dim = embeddings.shape[1]
        
        # Initialize or extend index
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine similarity)
        
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks with metadata
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_metadata.append({'source': source_file})
        
        self.save()
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[str]:
        """Tìm kiếm chunks tương tự nhất"""
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
        """Xóa toàn bộ index"""
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
    global _llm
    if _llm is None:
        from Llama_3_1_8B_Instruct_v2 import generate_response
        _llm = generate_response
        print(f"   ✅ Llama 3.1 8B loaded")
    return _llm


def generate_answer(query: str, context_chunks: List[Tuple[str, float]]) -> str:
    """Generate answer từ context chunks"""
    llm = get_llm()
    
    if not context_chunks:
        return "Tôi không tìm thấy thông tin liên quan trong tài liệu."
    
    # Build context
    context_parts = []
    for i, (chunk, score) in enumerate(context_chunks, 1):
        context_parts.append(f"[Đoạn {i}] (relevance: {score:.2f})\n{chunk}")
    
    context = "\n\n---\n\n".join(context_parts)
    
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
# RAG PIPELINE
# ═══════════════════════════════════════════════════════════
class RAGPro:
    def __init__(self):
        self.vector_store = VectorStore(RAG_STORAGE)
        self.tracker = IndexTracker(TRACKER_FILE)
    
    def index_file(self, file_path: str) -> int:
        """Index một file, trả về số chunks"""
        text = read_file(file_path)
        chunks = chunk_text(text)
        
        if chunks:
            self.vector_store.add_chunks(chunks, os.path.basename(file_path))
            self.tracker.mark_indexed(file_path, len(chunks))
        
        return len(chunks)
    
    def index_folder(self, folder: str, force: bool = False) -> int:
        """Index toàn bộ folder"""
        if force:
            self.vector_store.clear()
            self.tracker.indexed_files = {}
            self.tracker._save()
        
        if not os.path.exists(folder):
            print(f"⚠️ Folder không tồn tại: {folder}")
            return 0
        
        # Find files
        all_files = [f for f in os.listdir(folder) 
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        
        # Filter new files
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
        
        print(f"\n📊 Tổng: {total_chunks} chunks mới")
        return total_chunks
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Query RAG pipeline"""
        start = time.time()
        
        # Step 1: Retrieve from FAISS
        if verbose:
            print(f"   🔍 Searching...")
        retrieved_chunks = self.vector_store.search(question, TOP_K_RETRIEVE)
        
        if not retrieved_chunks:
            return "Tôi không tìm thấy thông tin liên quan trong database."
        
        if verbose:
            print(f"   📄 Found {len(retrieved_chunks)} chunks")
        
        # Step 2: Rerank
        if verbose:
            print(f"   🎯 Reranking...")
        reranked = rerank(question, retrieved_chunks, TOP_K_RERANK)
        
        if verbose:
            print(f"   ✅ Top {len(reranked)} chunks selected")
        
        # Step 3: Generate answer
        if verbose:
            print(f"   🤖 Generating answer...")
        answer = generate_answer(question, reranked)
        
        elapsed = time.time() - start
        if verbose:
            print(f"   ⏱️ Total: {elapsed:.1f}s")
        
        return answer
    
    def load(self) -> bool:
        """Load existing index"""
        return self.vector_store.load()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Pro - Optimized RAG System")
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index')
    parser.add_argument('--query', '-q', type=str, help='Single query mode')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG PRO - Phiên bản tối ưu")
    print("═" * 60)
    print(f"   📊 Embedding: {EMBEDDING_MODEL}")
    print(f"   🎯 Reranker:  {RERANKER_MODEL}")
    print(f"   🤖 LLM:       Llama 3.1 8B")
    print("═" * 60)
    
    # Initialize
    print("\n🔄 Loading models...")
    rag = RAGPro()
    
    # Load or create index
    if not args.force:
        rag.load()
    
    # Load all models
    get_embedder()
    get_reranker()
    get_llm()
    
    # Index
    print("\n" + "═" * 60)
    print("📁 INDEXING")
    print("═" * 60)
    rag.index_folder(COURSES_FOLDER, force=args.force)
    
    # Single query mode
    if args.query:
        print("\n" + "═" * 60)
        print("🔍 QUERY")
        print("═" * 60)
        print(f"\n❓ {args.query}")
        answer = rag.query(args.query)
        print(f"\n🤖 Answer:\n{answer}")
        return
    
    # Interactive mode
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
