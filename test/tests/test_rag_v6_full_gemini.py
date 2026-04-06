"""
Test RAG v6 - Complete Gemini Flow
- Index: Gemini API (chính xác hơn Local Llama)
- Query: Gemini API

Chạy: uv run test_rag_v6_full_gemini.py        # Index mới + Query
       uv run test_rag_v6_full_gemini.py --force  # Xóa và index lại

Yêu cầu: GEMINI_API_KEY trong file .env
"""

import asyncio
import json
import hashlib
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import LightRAG
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np

# ======================== CONFIG ========================
COURSES_FOLDER = "./courses"
RAG_STORAGE = "./rag_storage_gemini_v6"  # Database mới
INDEX_TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-pro"

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512


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
    
    def clear(self):
        self.indexed_files = {}
        self._save()


# ======================== FILE READERS ========================
def read_pdf_file(file_path: str) -> str:
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


# ======================== GEMINI LLM ========================
def create_gemini_llm():
    """Tạo Gemini LLM function"""
    from google import genai
    from google.genai import types
    
    if not GEMINI_API_KEY:
        raise ValueError("❌ Thiếu GEMINI_API_KEY trong .env")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    async def gemini_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        
        for msg in history_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                full_prompt += f"{role}: {content}\n"
        
        full_prompt += prompt
        
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            print(f"   ⚠️ Gemini error: {e}")
            return f"Error: {e}"
    
    return gemini_llm


# ======================== EMBEDDING ========================
def create_embedding():
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    async def embedding_func(texts):
        return embedder.encode(texts)
    
    return EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embedding_func
    )


# ======================== MAIN ========================
async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 TEST RAG v6 - Full Gemini Flow")
    print("   Index + Query đều dùng Gemini API")
    print("=" * 60)
    
    if not GEMINI_API_KEY:
        print("\n❌ Thiếu GEMINI_API_KEY!")
        print("   Thêm vào file .env: GEMINI_API_KEY=your_key")
        return
    
    print(f"✅ Gemini API Key: {GEMINI_API_KEY[:10]}...")
    
    # Force re-index?
    if args.force and os.path.exists(RAG_STORAGE):
        print(f"\n⚠️ Đang xóa database cũ: {RAG_STORAGE}")
        import shutil
        shutil.rmtree(RAG_STORAGE)
        print("   ✅ Đã xóa")
    
    os.makedirs(RAG_STORAGE, exist_ok=True)
    
    # Load models
    print("\n🔄 Loading models...")
    gemini_llm = create_gemini_llm()
    print("   ✅ Gemini LLM ready")
    embedding = create_embedding()
    print("   ✅ Embedding loaded")
    
    # Initialize LightRAG
    print("\n🔧 Initializing LightRAG...")
    rag = LightRAG(
        working_dir=RAG_STORAGE,
        llm_model_func=gemini_llm,
        embedding_func=embedding,
    )
    await rag.initialize_storages()
    print("✅ LightRAG ready!")
    
    # Auto-index
    print("\n" + "=" * 60)
    print("📁 KIỂM TRA FILE MỚI")
    print("=" * 60)
    
    tracker = IndexTracker(INDEX_TRACKER_FILE)
    
    if not os.path.exists(COURSES_FOLDER):
        print(f"⚠️ Folder {COURSES_FOLDER} không tồn tại")
    else:
        all_files = []
        for f in os.listdir(COURSES_FOLDER):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_files.append(f)
        
        new_files = [f for f in all_files 
                     if tracker.needs_indexing(os.path.join(COURSES_FOLDER, f))]
        
        if new_files:
            print(f"\n🆕 Phát hiện {len(new_files)} file cần index:")
            for f in new_files:
                print(f"   - {f}")
            
            print("\n📥 Đang index với Gemini...")
            for i, filename in enumerate(new_files, 1):
                file_path = os.path.join(COURSES_FOLDER, filename)
                try:
                    start = time.time()
                    text = read_file(file_path)
                    print(f"   [{i}/{len(new_files)}] {filename} ({len(text)} chars)...")
                    await rag.ainsert(text)
                    tracker.mark_indexed(file_path)
                    elapsed = time.time() - start
                    print(f"   [{i}/{len(new_files)}] ✅ {filename} ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"   [{i}/{len(new_files)}] ❌ {filename}: {e}")
        else:
            print(f"✅ Không có file mới. Database: {tracker.get_indexed_count()} files")
    
    # Interactive query
    print("\n" + "=" * 60)
    print("💬 CHẾ ĐỘ HỎI ĐÁP (Full Gemini)")
    print("=" * 60)
    print("Gõ câu hỏi. 'exit' để thoát.")
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
            
            print("🤖 Đang xử lý (Gemini)...")
            start = time.time()
            
            try:
                result = await rag.aquery(user_input, param=QueryParam(mode=current_mode))
                elapsed = time.time() - start
                print(f"\n🤖 AI ({elapsed:.1f}s):\n{result}")
            except Exception as e:
                print(f"❌ Lỗi: {e}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break


if __name__ == "__main__":
    asyncio.run(main())
