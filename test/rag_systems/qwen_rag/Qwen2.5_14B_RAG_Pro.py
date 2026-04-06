"""
RAG Pro + Qwen 2.5-14B - Phiên bản Vietnamese Optimized
═══════════════════════════════════════════════════════════
Architecture:
  1. Embedding:  bge-m3 (multilingual, best quality)
  2. Vector DB:  FAISS (fast, efficient)
  3. Reranker:   bge-reranker-v2-m3 (accuracy boost)
  4. LLM:        Qwen 2.5-14B (Vietnamese optimized, no Chinese output)

Chạy: uv run Qwen2.5_14B_RAG_Pro.py              # Index + Query
      uv run Qwen2.5_14B_RAG_Pro.py --force      # Re-index
      uv run Qwen2.5_14B_RAG_Pro.py --query "?"  # Single query
═══════════════════════════════════════════════════════════
"""

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

import json
import hashlib
import os
import time
import pickle
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
COURSES_FOLDER = "./courses"
RAG_STORAGE = "./rag_storage_qwen_pro"
INDEX_FILE = os.path.join(RAG_STORAGE, "faiss_index.pkl")
CHUNKS_FILE = os.path.join(RAG_STORAGE, "chunks.json")
TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Chunking config
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Model config
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QWEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# Retrieval config
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

# LLM config
LLM_MAX_TOKENS = 512


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
    if not chunks:
        return []
    
    reranker = get_reranker()
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    
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
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump(self.index, f)
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }, f, ensure_ascii=False, indent=2)
    
    def add_chunks(self, chunks: List[str], source_file: str):
        import faiss
        
        if not chunks:
            return
        
        embeddings = embed_texts(chunks)
        dim = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings.astype('float32'))
        
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_metadata.append({'source': source_file})
        
        self.save()
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Tuple[str, dict]]:
        """Tìm kiếm và trả về chunks với metadata"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = embed_query(query).reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], self.chunk_metadata[idx]))
        
        return results
    
    def get_stats(self) -> dict:
        """Lấy thống kê database"""
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'unique_sources': len(set(m.get('source', '') for m in self.chunk_metadata))
        }
    
    def clear(self):
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)


# ═══════════════════════════════════════════════════════════
# QWEN 2.5-14B LLM (Vietnamese Optimized)
# ═══════════════════════════════════════════════════════════
class QwenLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.bad_words_ids = None
    
    def load(self):
        if self.model is not None:
            return
        
        print(f"   📥 Loading {QWEN_MODEL}...")
        start = time.time()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        print(f"   ✅ Qwen loaded ({time.time()-start:.1f}s)")
        
        # Tạo bad_words_ids để chặn token ngoại ngữ
        print("   🔧 Tạo danh sách chặn token ngoại ngữ...")
        self.bad_words_ids = self._get_non_vietnamese_bad_words()
        print(f"   ✅ Đã chặn {len(self.bad_words_ids)} token ngoại ngữ!")
    
    def _get_non_vietnamese_bad_words(self):
        bad_words = []
        
        def is_allowed_char(ch):
            if ord(ch) < 128:
                return True
            if '\u00c0' <= ch <= '\u01b0':
                return True
            if '\u1ea0' <= ch <= '\u1ef9':
                return True
            if ch in '–—''""…•·×÷±≠≤≥':
                return True
            return False
        
        for i in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode([i])
            if any(not is_allowed_char(ch) for ch in token):
                bad_words.append([i])
        
        return bad_words
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if self.model is None:
            self.load()
        
        default_system = """Bạn là trợ lý AI chuyên giáo dục.
CHỈ trả lời bằng tiếng Việt.
Trả lời dựa trên thông tin được cung cấp.
Nếu không có thông tin, hãy nói thẳng."""
        
        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=LLM_MAX_TOKENS,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.2,
                bad_words_ids=self.bad_words_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][model_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response


# Global LLM instance
_llm = None

def get_llm() -> QwenLLM:
    global _llm
    if _llm is None:
        _llm = QwenLLM()
        _llm.load()
    return _llm


def generate_answer(query: str, context_chunks: List[Tuple[str, float, dict]]) -> Tuple[str, set]:
    """Generate answer và trả về (answer, source_files)"""
    llm = get_llm()
    
    if not context_chunks:
        return "Tôi không tìm thấy thông tin liên quan trong tài liệu.", set()
    
    # Collect sources
    sources = set()
    context_parts = []
    for i, (chunk, score, metadata) in enumerate(context_chunks, 1):
        source = metadata.get('source', 'unknown')
        sources.add(source)
        context_parts.append(f"[Đoạn {i} - {source}] (relevance: {score:.2f})\n{chunk}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Dựa trên ngữ cảnh sau, trả lời câu hỏi một cách chính xác.

QUY TẮC:
1. CHỈ sử dụng thông tin từ ngữ cảnh bên dưới
2. Nếu không có câu trả lời trong ngữ cảnh, nói "Tôi không tìm thấy thông tin này trong tài liệu."
3. Trả lời ngắn gọn, chính xác
4. CHỈ trả lời bằng tiếng Việt

NGỮ CẢNH:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""
    
    answer = llm.generate(prompt)
    return answer, sources


# ═══════════════════════════════════════════════════════════
# RAG PIPELINE
# ═══════════════════════════════════════════════════════════
class RAGPro:
    def __init__(self):
        self.vector_store = VectorStore(RAG_STORAGE)
        self.tracker = IndexTracker(TRACKER_FILE)
    
    def index_file(self, file_path: str) -> int:
        text = read_file(file_path)
        chunks = chunk_text(text)
        
        if chunks:
            self.vector_store.add_chunks(chunks, os.path.basename(file_path))
            self.tracker.mark_indexed(file_path, len(chunks))
        
        return len(chunks)
    
    def index_folder(self, folder: str, force: bool = False) -> int:
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
        
        print(f"\n📊 Tổng: {total_chunks} chunks mới")
        return total_chunks
    
    def query(self, question: str, verbose: bool = True) -> Tuple[str, set]:
        """Query và trả về (answer, source_files)"""
        start = time.time()
        
        if verbose:
            print(f"   🔍 Searching...")
        retrieved_chunks = self.vector_store.search(question, TOP_K_RETRIEVE)
        
        if not retrieved_chunks:
            return "Tôi không tìm thấy thông tin liên quan trong database.", set()
        
        if verbose:
            print(f"   📄 Found {len(retrieved_chunks)} chunks")
        
        # Extract chunks only for reranking
        chunks_only = [chunk for chunk, _ in retrieved_chunks]
        
        if verbose:
            print(f"   🎯 Reranking...")
        reranked = rerank(question, chunks_only, TOP_K_RERANK)
        
        if verbose:
            print(f"   ✅ Top {len(reranked)} chunks selected")
        
        # Add metadata back to reranked chunks
        reranked_with_metadata = []
        for chunk, score in reranked:
            # Find metadata for this chunk
            for orig_chunk, metadata in retrieved_chunks:
                if orig_chunk == chunk:
                    reranked_with_metadata.append((chunk, score, metadata))
                    break
        
        if verbose:
            print(f"   🤖 Generating answer...")
        answer, sources = generate_answer(question, reranked_with_metadata)
        
        elapsed = time.time() - start
        if verbose:
            print(f"   ⏱️ Total: {elapsed:.1f}s")
        
        return answer, sources
    
    def get_stats(self) -> dict:
        """Lấy thống kê hệ thống"""
        stats = self.vector_store.get_stats()
        stats['indexed_files'] = self.tracker.get_indexed_count()
        stats['total_chunks_tracked'] = self.tracker.get_total_chunks()
        return stats
    
    def load(self) -> bool:
        return self.vector_store.load()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Pro + Qwen 2.5-14B")
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index')
    parser.add_argument('--query', '-q', type=str, help='Single query mode')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG PRO + QWEN 2.5-14B (Vietnamese Optimized)")
    print("═" * 60)
    print(f"   📊 Embedding: {EMBEDDING_MODEL}")
    print(f"   🎯 Reranker:  {RERANKER_MODEL}")
    print(f"   🤖 LLM:       {QWEN_MODEL}")
    print("═" * 60)
    
    # Initialize
    print("\n🔄 Loading models...")
    rag = RAGPro()
    
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
        answer, sources = rag.query(args.query)
        print(f"\n🤖 Answer:\n{answer}")
        if sources:
            print(f"\n📚 Sources: {', '.join(sources)}")
        return
    
    # Interactive mode
    print("\n" + "═" * 60)
    print("💬 INTERACTIVE MODE")
    print("═" * 60)
    print("Commands: 'exit', 'stats', 'clear', 'help'")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🧑 Bạn: ").strip()
            
            if not question:
                continue
            
            # Commands
            if question.lower() in ["exit", "quit", "q", "thoát"]:
                print("👋 Tạm biệt!")
                break
            
            if question.lower() == "stats":
                stats = rag.get_stats()
                print("\n📊 Database Statistics:")
                print(f"   Files indexed: {stats['indexed_files']}")
                print(f"   Total chunks: {stats['total_chunks']}")
                print(f"   Unique sources: {stats['unique_sources']}")
                continue
            
            if question.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                print("═" * 60)
                print("💬 INTERACTIVE MODE")
                print("═" * 60)
                print("Commands: 'exit', 'stats', 'clear', 'help'")
                print("-" * 60)
                continue
            
            if question.lower() == "help":
                print("\n📖 Available Commands:")
                print("   exit/quit/q  - Thoát chương trình")
                print("   stats        - Hiển thị thống kê database")
                print("   clear        - Xóa màn hình")
                print("   help         - Hiển thị trợ giúp này")
                continue
            
            # Query
            print("\n🤖 Đang xử lý...")
            answer, sources = rag.query(question)
            print(f"\n📝 Trả lời:\n{answer}")
            if sources:
                print(f"\n📚 Nguồn: {', '.join(sorted(sources))}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")


if __name__ == "__main__":
    main()
