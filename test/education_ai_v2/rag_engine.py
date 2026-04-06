"""
RAG Engine - Xử lý tài liệu và trả lời câu hỏi
Sử dụng bge-m3 + bge-reranker + FAISS + Llama 3.1 8B
"""

import json
import hashlib
import os
import time
import pickle
import threading
from datetime import datetime
from typing import List, Tuple, Optional, Callable
from queue import Queue
from dataclasses import dataclass
from enum import Enum

import numpy as np


# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
STORAGE_DIR = "./rag_data"
UPLOADS_DIR = "./uploads"
SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Retrieval
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5


class FileStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class FileInfo:
    filename: str
    filepath: str
    status: FileStatus
    chunks: int = 0
    error: str = ""
    processed_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# FILE PROCESSOR
# ═══════════════════════════════════════════════════════════
class FileProcessor:
    """Xử lý file và chunk text"""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        import pdfplumber
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        return "\n\n".join(texts)
    
    @staticmethod
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
    
    @staticmethod
    def read_file(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return FileProcessor.read_pdf(file_path)
        return FileProcessor.read_text(file_path)
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


# ═══════════════════════════════════════════════════════════
# EMBEDDING & RERANKER (GPU Mode)
# ═══════════════════════════════════════════════════════════
class ModelManager:
    """Quản lý các models (singleton) - Embedding/Reranker chạy trên GPU"""
    
    _embedder = None
    _reranker = None
    _llm = None
    
    @classmethod
    def get_embedder(cls):
        if cls._embedder is None:
            from sentence_transformers import SentenceTransformer
            print("📥 Loading Embedder on GPU...")
            cls._embedder = SentenceTransformer(EMBEDDING_MODEL)
            print("✅ Embedder ready")
        return cls._embedder
    
    @classmethod
    def get_reranker(cls):
        if cls._reranker is None:
            from sentence_transformers import CrossEncoder
            print("📥 Loading Reranker on GPU...")
            cls._reranker = CrossEncoder(RERANKER_MODEL)
            print("✅ Reranker ready")
        return cls._reranker
    
    @classmethod
    def get_llm(cls):
        if cls._llm is None:
            from engine_llm import EducationalLLM
            cls._llm = EducationalLLM()
        return cls._llm
    
    @classmethod
    def embed_texts(cls, texts: List[str]) -> np.ndarray:
        embedder = cls.get_embedder()
        return embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    @classmethod
    def embed_query(cls, query: str) -> np.ndarray:
        embedder = cls.get_embedder()
        return embedder.encode([query], normalize_embeddings=True)[0]
    
    @classmethod
    def rerank(cls, query: str, chunks: List[str], top_k: int = TOP_K_RERANK) -> List[Tuple[str, float]]:
        if not chunks:
            return []
        reranker = cls.get_reranker()
        pairs = [(query, chunk) for chunk in chunks]
        scores = reranker.predict(pairs)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]


# ═══════════════════════════════════════════════════════════
# VECTOR STORE
# ═══════════════════════════════════════════════════════════
class VectorStore:
    """FAISS Vector Store"""
    
    def __init__(self, storage_dir: str = STORAGE_DIR):
        self.storage_dir = storage_dir
        self.index_file = os.path.join(storage_dir, "faiss_index.pkl")
        self.chunks_file = os.path.join(storage_dir, "chunks.json")
        self.tracker_file = os.path.join(storage_dir, "files.json")
        
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.files = {}
        
        os.makedirs(storage_dir, exist_ok=True)
        self.load()
    
    def load(self) -> bool:
        try:
            if os.path.exists(self.index_file):
                import faiss
                with open(self.index_file, 'rb') as f:
                    self.index = pickle.load(f)
            
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = data.get('chunks', [])
                    self.chunk_metadata = data.get('metadata', [])
            
            if os.path.exists(self.tracker_file):
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    self.files = json.load(f)
            
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False
    
    def save(self):
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)
        
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }, f, ensure_ascii=False, indent=2)
        
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.files, f, ensure_ascii=False, indent=2)
    
    def add_file(self, filepath: str, filename: str) -> int:
        """Thêm file vào index, trả về số chunks"""
        import faiss
        
        # Read and chunk
        text = FileProcessor.read_file(filepath)
        chunks = FileProcessor.chunk_text(text)
        
        if not chunks:
            return 0
        
        # Embed
        embeddings = ModelManager.embed_texts(chunks)
        dim = embeddings.shape[1]
        
        # Initialize or extend index
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_metadata.append({'source': filename})
        
        # Track file
        self.files[filename] = {
            'hash': FileProcessor.get_file_hash(filepath),
            'chunks': len(chunks),
            'indexed_at': datetime.now().isoformat()
        }
        
        self.save()
        return len(chunks)
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[str]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = ModelManager.embed_query(query).reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results
    
    def is_file_indexed(self, filepath: str, filename: str) -> bool:
        if filename not in self.files:
            return False
        current_hash = FileProcessor.get_file_hash(filepath)
        return self.files[filename].get('hash') == current_hash
    
    def get_stats(self) -> dict:
        return {
            'files': len(self.files),
            'chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0
        }


# ═══════════════════════════════════════════════════════════
# RAG ENGINE
# ═══════════════════════════════════════════════════════════
class RAGEngine:
    """RAG Engine chính"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.file_queue: List[FileInfo] = []
        self.processing = False
        self._status_callback: Optional[Callable] = None
        
        os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    def set_status_callback(self, callback: Callable):
        """Set callback để cập nhật UI"""
        self._status_callback = callback
    
    def _update_status(self):
        if self._status_callback:
            self._status_callback(self.file_queue)
    
    def add_file(self, filepath: str, filename: str) -> FileInfo:
        """Thêm file vào queue"""
        info = FileInfo(
            filename=filename,
            filepath=filepath,
            status=FileStatus.PENDING
        )
        self.file_queue.append(info)
        self._update_status()
        return info
    
    def process_queue(self):
        """Xử lý tất cả file trong queue"""
        for info in self.file_queue:
            if info.status == FileStatus.PENDING:
                self._process_file(info)
    
    def _process_file(self, info: FileInfo):
        """Xử lý một file"""
        try:
            info.status = FileStatus.PROCESSING
            self._update_status()
            
            # Check if already indexed
            if self.vector_store.is_file_indexed(info.filepath, info.filename):
                info.status = FileStatus.COMPLETED
                info.chunks = self.vector_store.files.get(info.filename, {}).get('chunks', 0)
                info.processed_at = datetime.now().isoformat()
                self._update_status()
                return
            
            # Process
            chunks = self.vector_store.add_file(info.filepath, info.filename)
            
            info.status = FileStatus.COMPLETED
            info.chunks = chunks
            info.processed_at = datetime.now().isoformat()
            
        except Exception as e:
            info.status = FileStatus.ERROR
            info.error = str(e)
        
        self._update_status()
    
    def query(self, question: str) -> Tuple[str, List[Tuple[str, float]]]:
        """Query RAG và trả về (answer, context_chunks)"""
        # Retrieve
        retrieved = self.vector_store.search(question, TOP_K_RETRIEVE)
        
        if not retrieved:
            return "Tôi không tìm thấy thông tin liên quan trong tài liệu đã upload.", []
        
        # Rerank
        reranked = ModelManager.rerank(question, retrieved, TOP_K_RERANK)
        
        return reranked
    
    def get_file_status(self) -> List[dict]:
        """Lấy trạng thái của tất cả files"""
        return [
            {
                'filename': f.filename,
                'status': f.status.value,
                'chunks': f.chunks,
                'error': f.error,
                'processed_at': f.processed_at
            }
            for f in self.file_queue
        ]
    
    def get_stats(self) -> dict:
        return self.vector_store.get_stats()


# Singleton instance
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
