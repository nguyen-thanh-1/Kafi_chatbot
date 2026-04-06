"""
Test RAG v4 - LightRAG Complete Flow
Kết hợp Indexing + Querying trong 1 file duy nhất
Tự động phát hiện và index file mới

Chạy: uv run test_rag_v4_lightrag.py
"""

import asyncio
import json
import hashlib
import os
import time
from datetime import datetime

# Import LightRAG
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np

# ======================== CONFIG ========================
COURSES_FOLDER = "./courses"
RAG_STORAGE = "./rag_storage_v4"
INDEX_TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512

# LLM
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.1


# ======================== GLOBAL VARIABLES ========================
_llm_func = None
_embedding_func = None
_rag_instance = None


# ======================== INDEX TRACKER ========================
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
    
    def mark_indexed(self, file_path: str):
        filename = os.path.basename(file_path)
        self.indexed_files[filename] = {
            'hash': self._get_file_hash(file_path),
            'indexed_at': datetime.now().isoformat(),
            'size_bytes': os.path.getsize(file_path),
        }
        self._save()
    
    def get_indexed_count(self) -> int:
        return len(self.indexed_files)


# ======================== FILE READERS ========================
def read_pdf_file(file_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        import subprocess
        subprocess.run(["uv", "pip", "install", "pdfplumber"], check=True)
        import pdfplumber
    
    text_content = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
    return "\n\n".join(text_content)


def read_text_file(file_path: str) -> str:
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')


def read_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return read_pdf_file(file_path)
    return read_text_file(file_path)


# ======================== MODEL SETUP ========================
def load_models():
    """Load models một lần duy nhất"""
    global _llm_func, _embedding_func
    
    if _llm_func is not None and _embedding_func is not None:
        return _llm_func, _embedding_func
    
    print("🔄 Loading models...")
    start = time.time()
    
    # Load Llama
    from Llama_3_1_8B_Instruct_v2 import generate_response
    
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return generate_response(
            user_input=prompt,
            history=history_messages or [],
            system_prompt=system_prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE
        )
    
    _llm_func = llm_func
    print(f"   ✅ Llama 3.1 8B loaded ({time.time()-start:.1f}s)")
    
    # Load Embedding
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    async def embedding_func(texts):
        return embedder.encode(texts)
    
    _embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embedding_func
    )
    print(f"   ✅ Embedding loaded")
    
    return _llm_func, _embedding_func


async def get_rag_instance():
    """Lấy hoặc tạo RAG instance"""
    global _rag_instance
    
    if _rag_instance is not None:
        return _rag_instance
    
    os.makedirs(RAG_STORAGE, exist_ok=True)
    
    llm_func, embedding_func = load_models()
    
    print("🔧 Initializing LightRAG...")
    _rag_instance = LightRAG(
        working_dir=RAG_STORAGE,
        llm_model_func=llm_func,
        embedding_func=embedding_func,
    )
    await _rag_instance.initialize_storages()
    print("✅ LightRAG ready!")
    
    return _rag_instance


# ======================== AUTO INDEX ========================
async def auto_index_new_files():
    """Tự động phát hiện và index file mới"""
    rag = await get_rag_instance()
    tracker = IndexTracker(INDEX_TRACKER_FILE)
    
    # Tìm files
    if not os.path.exists(COURSES_FOLDER):
        print(f"⚠️ Folder {COURSES_FOLDER} không tồn tại")
        return 0
    
    all_files = []
    for f in os.listdir(COURSES_FOLDER):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            all_files.append(f)
    
    # Lọc files mới
    new_files = []
    for f in all_files:
        file_path = os.path.join(COURSES_FOLDER, f)
        if tracker.needs_indexing(file_path):
            new_files.append(f)
    
    if not new_files:
        print(f"✅ Không có file mới. Database: {tracker.get_indexed_count()} files")
        return 0
    
    print(f"\n🆕 Phát hiện {len(new_files)} file mới:")
    for f in new_files:
        print(f"   - {f}")
    
    # Index
    print("\n📥 Đang index...")
    indexed = 0
    
    for i, filename in enumerate(new_files, 1):
        file_path = os.path.join(COURSES_FOLDER, filename)
        
        try:
            start = time.time()
            text = read_file(file_path)
            await rag.ainsert(text)
            tracker.mark_indexed(file_path)
            elapsed = time.time() - start
            print(f"   [{i}/{len(new_files)}] ✅ {filename} ({elapsed:.1f}s)")
            indexed += 1
        except Exception as e:
            print(f"   [{i}/{len(new_files)}] ❌ {filename}: {e}")
    
    print(f"\n📊 Đã index: {indexed}/{len(new_files)} files")
    return indexed


# ======================== QUERY ========================
async def query(question: str, mode: str = "hybrid") -> str:
    """Query RAG"""
    rag = await get_rag_instance()
    
    try:
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        return result
    except Exception as e:
        return f"❌ Lỗi: {e}"


# ======================== INTERACTIVE MODE ========================
async def interactive_mode():
    """Chế độ hỏi đáp tương tác"""
    print("\n" + "=" * 60)
    print("💬 CHẾ ĐỘ HỎI ĐÁP")
    print("=" * 60)
    print("Gõ câu hỏi và Enter. 'exit' để thoát.")
    print("'mode:hybrid/local/global/naive' để đổi mode")
    print("-" * 60)
    
    current_mode = "hybrid"
    
    while True:
        try:
            user_input = input(f"\n🧑 [{current_mode}] Bạn: ").strip()
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break
            
            if not user_input:
                continue
            
            if user_input.startswith("mode:"):
                new_mode = user_input.split(":")[1].strip()
                if new_mode in ["hybrid", "local", "global", "naive"]:
                    current_mode = new_mode
                    print(f"✅ Mode: {current_mode}")
                continue
            
            print("🤖 Đang xử lý...")
            start = time.time()
            result = await query(user_input, mode=current_mode)
            elapsed = time.time() - start
            print(f"\n🤖 AI ({elapsed:.1f}s):\n{result}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break


# ======================== MAIN ========================
async def main():
    print("=" * 60)
    print("🚀 TEST RAG v4 - LightRAG Complete Flow")
    print("=" * 60)
    
    total_start = time.time()
    
    # 1. Khởi tạo và load models
    await get_rag_instance()
    
    # 2. Auto-index new files
    print("\n" + "=" * 60)
    print("📁 KIỂM TRA FILE MỚI")
    print("=" * 60)
    await auto_index_new_files()
    
    # 3. Test queries
    print("\n" + "=" * 60)
    print("🧪 TEST QUERIES")
    print("=" * 60)
    
    test_questions = [
        "RAG là gì?",
        "Những loại Machine Learning nào được đề cập?",
    ]
    
    for q in test_questions:
        print(f"\n❓ {q}")
        start = time.time()
        answer = await query(q, mode="hybrid")
        elapsed = time.time() - start
        print(f"📝 ({elapsed:.1f}s): {answer[:300]}..." if len(answer) > 300 else f"📝 ({elapsed:.1f}s): {answer}")
    
    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"⏱️ TỔNG THỜI GIAN: {total_elapsed:.1f}s")
    print("=" * 60)
    
    # 4. Interactive mode
    print("\n💡 Bạn có muốn vào chế độ hỏi đáp không? (y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
