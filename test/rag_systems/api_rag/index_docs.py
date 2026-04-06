"""
Index Documents - Script index tài liệu (chạy offline)
Sử dụng RAGAnything để parse và index tài liệu vào vector database

Chạy: uv run index_docs.py
Hoặc: uv run index_docs.py --force  (để re-index tất cả)
"""

import asyncio
import argparse
import json
import hashlib
import os
from datetime import datetime

# Import RAGAnything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# Import config
from rag_config import (
    COURSES_FOLDER, OUTPUT_DIR, RAG_STORAGE, INDEX_TRACKER_FILE,
    SUPPORTED_EXTENSIONS, EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    RAG_CONFIG, ensure_directories, get_supported_files, get_file_info
)


# ======================== INDEX TRACKER ========================
class IndexTracker:
    """Quản lý danh sách file đã index (cho incremental indexing)"""
    
    def __init__(self, tracker_file: str):
        self.tracker_file = tracker_file
        self.indexed_files = self._load()
    
    def _load(self) -> dict:
        """Load tracker từ file"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save(self):
        """Lưu tracker vào file"""
        os.makedirs(os.path.dirname(self.tracker_file), exist_ok=True)
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.indexed_files, f, indent=2, ensure_ascii=False)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Tính hash của file để detect thay đổi"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Đọc theo chunks để tránh dùng quá nhiều RAM với file lớn
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def needs_indexing(self, file_path: str) -> bool:
        """Kiểm tra file có cần index không (mới hoặc đã thay đổi)"""
        if not os.path.exists(file_path):
            return False
        
        filename = os.path.basename(file_path)
        current_hash = self._get_file_hash(file_path)
        
        if filename not in self.indexed_files:
            return True  # File mới
        
        if self.indexed_files[filename].get('hash') != current_hash:
            return True  # File đã thay đổi
        
        return False  # File không đổi
    
    def mark_indexed(self, file_path: str):
        """Đánh dấu file đã được index"""
        filename = os.path.basename(file_path)
        self.indexed_files[filename] = {
            'hash': self._get_file_hash(file_path),
            'indexed_at': datetime.now().isoformat(),
            'size_bytes': os.path.getsize(file_path),
        }
        self._save()
    
    def get_indexed_count(self) -> int:
        """Số file đã index"""
        return len(self.indexed_files)


# ======================== MODELS SETUP ========================
def setup_models():
    """Load và setup các models (embedding, LLM)"""
    print("🔄 Loading models...")
    
    # Import Llama model
    try:
        from Llama_3_1_8B_Instruct_v2 import generate_response
        print("   ✅ Llama 3.1 8B loaded")
    except ImportError as e:
        print(f"   ❌ Không thể load Llama model: {e}")
        raise
    
    # Import sentence_transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("   ❌ Chưa cài sentence-transformers")
        raise
    
    # Load embedding model
    print(f"   Loading embedding: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("   ✅ Embedding model loaded")
    
    # Create async embedding function
    async def embedding_func(texts):
        return embedder.encode(texts)
    
    embedding = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embedding_func
    )
    
    # Create async LLM function
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        print(f"   🤖 LLM processing ({len(prompt)} chars)...")
        chat_history = history_messages if history_messages else []
        response = generate_response(
            user_input=prompt,
            history=chat_history,
            system_prompt=system_prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE
        )
        return response
    
    return embedding, llm_func


# ======================== MAIN INDEXING ========================
async def index_documents(force_reindex: bool = False):
    """Index tất cả tài liệu trong folder"""
    
    print("=" * 60)
    print("📚 RAG DOCUMENT INDEXER")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_directories()
    
    # Setup tracker
    tracker = IndexTracker(INDEX_TRACKER_FILE)
    print(f"📊 Đã có {tracker.get_indexed_count()} file(s) trong database")
    
    # Get files to process
    all_files = get_supported_files(COURSES_FOLDER)
    if not all_files:
        print(f"❌ Không tìm thấy file nào trong {COURSES_FOLDER}")
        print(f"   Định dạng hỗ trợ: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"📁 Tìm thấy {len(all_files)} file(s) trong {COURSES_FOLDER}")
    
    # Determine which files need indexing
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
            print("   Dùng --force để re-index tất cả.")
            return
        
        print(f"🆕 {len(files_to_index)} file(s) cần index (mới hoặc đã thay đổi)")
    
    # Show files to be indexed
    print("\n📋 Files sẽ được index:")
    for i, f in enumerate(files_to_index, 1):
        info = get_file_info(COURSES_FOLDER, f)
        print(f"   {i}. {f} ({info['size_mb']:.2f} MB)")
    
    # Setup models
    print()
    embedding, llm_func = setup_models()
    
    # Initialize RAG
    print("\n🔧 Initializing RAGAnything...")
    config = RAGAnythingConfig(**RAG_CONFIG)
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func, # llm_func
        embedding_func=embedding,
    )
    print("✅ RAGAnything initialized")
    
    # Index each file
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
    print("📊 KẾT QUẢ INDEXING")
    print("=" * 60)
    print(f"   ✅ Thành công: {success_count} file(s)")
    print(f"   ❌ Lỗi: {error_count} file(s)")
    print(f"   📦 Tổng trong database: {tracker.get_indexed_count()} file(s)")
    print(f"\n💾 Database lưu tại: {RAG_STORAGE}")
    print("🚀 Chạy query_rag.py để bắt đầu hỏi đáp!")


# ======================== CLI ========================
def main():
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-index tất cả files (bỏ qua cache)'
    )
    args = parser.parse_args()
    
    asyncio.run(index_documents(force_reindex=args.force))


if __name__ == "__main__":
    main()
