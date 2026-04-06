"""
Index Documents với Gemini API - Miễn phí, limit cao
Sử dụng Gemini API cho entity extraction

Chạy: uv run index_docs_gemini.py
Hoặc: uv run index_docs_gemini.py --force

Yêu cầu: Đặt GEMINI_API_KEY trong file .env
"""

import asyncio
import argparse
import json
import hashlib
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Import RAGAnything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# ======================== GEMINI CONFIG ========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Gemini 2.0 Flash: nhanh, miễn phí, 1500 requests/ngày
GEMINI_MODEL = "gemini-2.0-flash"

# ======================== PATHS (RIÊNG BIỆT) ========================
COURSES_FOLDER = "./courses"
OUTPUT_DIR = "./output_courses_gemini"  # Output riêng
RAG_STORAGE = "./rag_storage_gemini"  # Storage riêng
INDEX_TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pptx", ".ppt", ".md"]

# Embedding config (vẫn dùng local)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 256

# Retry config
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds


# ======================== INDEX TRACKER ========================
class IndexTracker:
    """Quản lý danh sách file đã index"""
    
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
            'method': 'gemini'
        }
        self._save()
    
    def get_indexed_count(self) -> int:
        return len(self.indexed_files)


# ======================== GEMINI LLM FUNCTION ========================
def create_gemini_llm_func():
    """Tạo async LLM function sử dụng Gemini API"""
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Chưa cài google-generativeai. Chạy: uv pip install google-generativeai")
        raise
    
    if not GEMINI_API_KEY:
        raise ValueError(
            "❌ Chưa có GEMINI_API_KEY!\n"
            "   1. Vào: https://aistudio.google.com/app/apikey\n"
            "   2. Tạo API Key\n"
            "   3. Thêm vào file .env: GEMINI_API_KEY=your_key_here"
        )
    
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    async def gemini_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        """Gọi Gemini API với retry"""
        
        # Build prompt with system instruction
        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n"
        
        # Add history
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
        
        full_prompt += f"User: {prompt}"
        
        # Retry with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4096,
                    )
                )
                return response.text
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = INITIAL_BACKOFF * (2 ** attempt)
                        print(f"   ⏳ Rate limit, đợi {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ Rate limit sau {MAX_RETRIES} lần thử")
                        raise
                else:
                    print(f"   ⚠️ Gemini API error: {e}")
                    raise
    
    return gemini_llm_func


# ======================== EMBEDDING FUNCTION ========================
def create_embedding_func():
    """Tạo embedding function (vẫn dùng local)"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ Chưa cài sentence-transformers")
        raise
    
    print(f"   Loading embedding: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("   ✅ Embedding model loaded")
    
    async def embedding_func(texts):
        return embedder.encode(texts)
    
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embedding_func
    )


# ======================== HELPER FUNCTIONS ========================
def ensure_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    """Index tài liệu với Gemini API"""
    
    print("=" * 60)
    print("🚀 RAG INDEXER với GEMINI API")
    print("=" * 60)
    
    # Check API key
    if not GEMINI_API_KEY:
        print("\n❌ Thiếu GEMINI_API_KEY!")
        print("   1. Vào: https://aistudio.google.com/app/apikey")
        print("   2. Tạo API Key")
        print("   3. Thêm vào file .env: GEMINI_API_KEY=your_key_here")
        return
    
    print(f"✅ Gemini API Key: {GEMINI_API_KEY[:10]}...")
    print(f"📦 Model: {GEMINI_MODEL}")
    
    # Ensure directories
    ensure_directories()
    
    print(f"\n📁 Cấu hình:")
    print(f"   - Tài liệu: {COURSES_FOLDER}")
    print(f"   - Output: {OUTPUT_DIR}")
    print(f"   - Database: {RAG_STORAGE}")
    
    # Setup tracker
    tracker = IndexTracker(INDEX_TRACKER_FILE)
    print(f"📊 Đã có {tracker.get_indexed_count()} file(s) trong database")
    
    # Get files
    all_files = get_supported_files(COURSES_FOLDER)
    if not all_files:
        print(f"\n❌ Không tìm thấy file trong {COURSES_FOLDER}")
        return
    
    print(f"📁 Tìm thấy {len(all_files)} file(s)")
    
    # Determine files to index
    if force_reindex:
        files_to_index = all_files
        print("⚠️  Force re-index: Sẽ index lại TẤT CẢ files")
    else:
        files_to_index = []
        for f in all_files:
            file_path = os.path.join(COURSES_FOLDER, f)
            if tracker.needs_indexing(file_path):
                files_to_index.append(f)
        
        if not files_to_index:
            print("✅ Tất cả files đã được index. Không có gì mới.")
            return
        
        print(f"🆕 {len(files_to_index)} file(s) cần index")
    
    # Show files
    print("\n📋 Files sẽ được index:")
    for i, f in enumerate(files_to_index, 1):
        info = get_file_info(COURSES_FOLDER, f)
        print(f"   {i}. {f} ({info['size_mb']:.2f} MB)")
    
    # Setup models
    print("\n🔄 Loading models...")
    gemini_llm = create_gemini_llm_func()
    print("   ✅ Gemini LLM ready")
    embedding_func = create_embedding_func()
    
    # Initialize RAG
    print("\n🔧 Initializing RAGAnything với Gemini...")
    config = RAGAnythingConfig(
        working_dir=RAG_STORAGE,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=False,
    )
    
    rag = RAGAnything(
        config=config,
        llm_model_func=gemini_llm,
        embedding_func=embedding_func,
    )
    print("✅ RAGAnything initialized với Gemini")
    
    # Index files
    print("\n" + "=" * 60)
    print("🚀 BẮT ĐẦU INDEXING (Gemini API)")
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
            
            await rag.process_document_complete(
                file_path=file_path,
                output_dir=OUTPUT_DIR,
                parse_method="auto"
            )
            
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
    print("📊 KẾT QUẢ INDEXING (Gemini)")
    print("=" * 60)
    print(f"   ✅ Thành công: {success_count} file(s)")
    print(f"   ❌ Lỗi: {error_count} file(s)")
    print(f"   📦 Tổng trong database: {tracker.get_indexed_count()} file(s)")
    print(f"\n💾 Database lưu tại: {RAG_STORAGE}")
    print("🚀 Chạy 'uv run query_rag_gemini.py' để hỏi đáp!")


# ======================== CLI ========================
def main():
    parser = argparse.ArgumentParser(description="Index với Gemini API")
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index')
    args = parser.parse_args()
    
    asyncio.run(index_documents(force_reindex=args.force))


if __name__ == "__main__":
    main()
