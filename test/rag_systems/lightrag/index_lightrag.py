"""
Index Documents với LightRAG trực tiếp + Local Llama 3.1 8B
Nhanh hơn RAGAnything, không cần API key

Chạy: uv run index_lightrag.py
Hoặc: uv run index_lightrag.py --force
"""

import asyncio
import argparse
import json
import hashlib
import os
from datetime import datetime

# Import LightRAG trực tiếp
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np

# ======================== CONFIG ========================
# Paths
COURSES_FOLDER = "./courses"
RAG_STORAGE = "./rag_storage_lightrag"
INDEX_TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

# Supported file types
SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Embedding config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512

# LLM config
LLM_MAX_NEW_TOKENS = 2048
LLM_TEMPERATURE = 0.1


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


# ======================== LOCAL LLM FUNCTION ========================
def create_local_llm_func():
    """Tạo LLM function cho LightRAG sử dụng Local Llama 3.1 8B"""
    try:
        from Llama_3_1_8B_Instruct_v2 import generate_response
        print("   ✅ Llama 3.1 8B loaded")
    except ImportError as e:
        print(f"❌ Không thể import Llama model: {e}")
        raise
    
    async def local_llm_complete(
        prompt, 
        system_prompt=None, 
        history_messages=[], 
        **kwargs
    ) -> str:
        """Local Llama completion function cho LightRAG"""
        chat_history = history_messages if history_messages else []
        
        response = generate_response(
            user_input=prompt,
            history=chat_history,
            system_prompt=system_prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE
        )
        return response
    
    return local_llm_complete


# ======================== EMBEDDING FUNCTION ========================
def create_embedding_func():
    """Tạo embedding function cho LightRAG"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Chưa cài sentence-transformers")
        raise
    
    print(f"   Loading embedding: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("   ✅ Embedding loaded")
    
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return embedder.encode(texts)
    
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embedding_func
    )


# ======================== READ FILES ========================
def read_pdf_file(file_path: str) -> str:
    """Đọc nội dung PDF file bằng pdfplumber"""
    try:
        import pdfplumber
    except ImportError:
        print("   ⚠️ Đang cài pdfplumber...")
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
    """Đọc nội dung text file"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    # Fallback: binary read and decode
    with open(file_path, 'rb') as f:
        content = f.read()
        return content.decode('utf-8', errors='ignore')


def read_file(file_path: str) -> str:
    """Đọc file dựa vào extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return read_pdf_file(file_path)
    else:
        return read_text_file(file_path)


# ======================== HELPERS ========================
def ensure_directories():
    os.makedirs(RAG_STORAGE, exist_ok=True)


def get_supported_files(folder: str) -> list:
    if not os.path.exists(folder):
        return []
    files = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def get_file_info(folder: str, filename: str) -> dict:
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        return None
    stat = os.stat(file_path)
    return {
        "filename": filename,
        "path": file_path,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
    }


# ======================== MAIN INDEXING ========================
async def index_documents(force_reindex: bool = False):
    """Index tài liệu với LightRAG + Local Llama"""
    
    print("=" * 60)
    print("🚀 LightRAG INDEXER (Local Llama 3.1 8B)")
    print("   Không cần API key - chạy hoàn toàn local")
    print("=" * 60)
    
    ensure_directories()
    
    print(f"\n📁 Cấu hình:")
    print(f"   - Tài liệu: {COURSES_FOLDER}")
    print(f"   - Database: {RAG_STORAGE}")
    print(f"   - Formats: {SUPPORTED_EXTENSIONS}")
    
    # Tracker
    tracker = IndexTracker(INDEX_TRACKER_FILE)
    print(f"📊 Đã có {tracker.get_indexed_count()} file(s) trong database")
    
    # Get files
    all_files = get_supported_files(COURSES_FOLDER)
    if not all_files:
        print(f"\n❌ Không tìm thấy file text trong {COURSES_FOLDER}")
        print(f"   Chỉ hỗ trợ: {SUPPORTED_EXTENSIONS}")
        print(f"   Nếu có PDF, hãy convert sang text trước")
        return
    
    print(f"📁 Tìm thấy {len(all_files)} file(s)")
    
    # Files to index
    if force_reindex:
        files_to_index = all_files
        print("⚠️  Force re-index: TẤT CẢ files")
    else:
        files_to_index = []
        for f in all_files:
            file_path = os.path.join(COURSES_FOLDER, f)
            if tracker.needs_indexing(file_path):
                files_to_index.append(f)
        
        if not files_to_index:
            print("✅ Tất cả files đã được index.")
            return
        
        print(f"🆕 {len(files_to_index)} file(s) cần index")
    
    # Show files
    print("\n📋 Files:")
    for i, f in enumerate(files_to_index, 1):
        info = get_file_info(COURSES_FOLDER, f)
        print(f"   {i}. {f} ({info['size_mb']:.2f} MB)")
    
    # Setup models
    print("\n🔄 Loading models...")
    local_llm = create_local_llm_func()
    embedding_func = create_embedding_func()
    
    # Initialize LightRAG
    print("\n🔧 Initializing LightRAG...")
    rag = LightRAG(
        working_dir=RAG_STORAGE,
        llm_model_func=local_llm,
        embedding_func=embedding_func,
    )
    
    # QUAN TRỌNG: Phải initialize storages trước khi dùng
    await rag.initialize_storages()
    print("✅ LightRAG initialized")
    
    # Index
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU INDEXING")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for i, filename in enumerate(files_to_index, 1):
        file_path = os.path.join(COURSES_FOLDER, filename)
        info = get_file_info(COURSES_FOLDER, filename)
        
        print(f"\n[{i}/{len(files_to_index)}] 📄 {filename}")
        print(f"    Size: {info['size_mb']:.2f} MB")
        
        try:
            start_time = datetime.now()
            
            # Đọc file content (hỗ trợ PDF và text)
            text_content = read_file(file_path)
            print(f"    📝 {len(text_content)} characters")
            
            # Insert vào LightRAG
            await rag.ainsert(text_content)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            tracker.mark_indexed(file_path)
            
            print(f"    ✅ Hoàn thành trong {elapsed:.1f}s")
            success_count += 1
            
        except Exception as e:
            print(f"    ❌ Lỗi: {str(e)}")
            error_count += 1
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ")
    print("=" * 60)
    print(f"   ✅ Thành công: {success_count}")
    print(f"   ❌ Lỗi: {error_count}")
    print(f"   📦 Tổng: {tracker.get_indexed_count()}")
    print(f"\n💾 Database: {RAG_STORAGE}")
    print("🚀 Chạy 'uv run query_lightrag.py' để hỏi đáp!")


def main():
    parser = argparse.ArgumentParser(description="LightRAG Indexer")
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()
    asyncio.run(index_documents(force_reindex=args.force))


if __name__ == "__main__":
    main()
