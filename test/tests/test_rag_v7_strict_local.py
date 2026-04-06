"""
Test RAG v7 - Local Llama với System Prompt mạnh
Cải thiện độ chính xác bằng cách:
1. Dùng mode "naive" - truy xuất chunk trực tiếp
2. System prompt bắt buộc tuân theo context
3. Không cho phép hallucinate

Chạy: uv run test_rag_v7_strict_local.py
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
RAG_STORAGE = "./rag_storage_v7"  # Database mới
INDEX_TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512

# LLM Config
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.05  # Rất thấp để giảm hallucination

# System prompt BẮT BUỘC tuân theo context
STRICT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions ONLY based on the provided context.

RULES:
1. ONLY use information from the context provided below
2. If the answer is NOT in the context, say "Tôi không tìm thấy thông tin này trong tài liệu."
3. DO NOT make up or hallucinate any information
4. Quote specific parts of the context when answering
5. Be concise and factual

CONTEXT:
{context}

Based ONLY on the above context, answer the following question:"""


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


# ======================== LOCAL LLM ========================
_generate_response = None

def load_llm():
    global _generate_response
    if _generate_response is None:
        from Llama_3_1_8B_Instruct_v2 import generate_response
        _generate_response = generate_response
    return _generate_response


def create_local_llm():
    """Tạo Local LLM function với system prompt mạnh"""
    generate_response = load_llm()
    
    async def local_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
        # Luôn dùng system prompt strict
        final_prompt = prompt
        
        response = generate_response(
            user_input=final_prompt,
            history=[],
            system_prompt=system_prompt or "You are a helpful assistant.",
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        return response
    
    return local_llm


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


# ======================== DIRECT SEARCH (KHÔNG DÙNG RAG GRAPH) ========================
async def direct_search(query: str, rag_storage: str, top_k: int = 5):
    """Tìm kiếm trực tiếp trong text chunks"""
    chunks_file = os.path.join(rag_storage, "kv_store_text_chunks.json")
    
    if not os.path.exists(chunks_file):
        return []
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Simple keyword matching
    query_lower = query.lower()
    query_words = query_lower.split()
    
    scored_chunks = []
    for chunk_id, chunk_info in chunks_data.items():
        content = chunk_info.get('content', '')
        content_lower = content.lower()
        
        # Score based on keyword matches
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored_chunks.append((score, content))
    
    # Sort by score and return top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk[1] for chunk in scored_chunks[:top_k]]


async def answer_with_context(query: str, context_chunks: list, llm_func):
    """Trả lời câu hỏi dựa trên context"""
    if not context_chunks:
        return "Tôi không tìm thấy thông tin liên quan trong tài liệu."
    
    context = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "Tôi không tìm thấy thông tin này trong tài liệu."

CONTEXT:
{context}

QUESTION: {query}

ANSWER (based ONLY on the context above):"""
    
    response = await llm_func(prompt, system_prompt="You are a helpful assistant that answers questions ONLY based on the provided context. Never make up information.")
    return response


# ======================== MAIN ========================
async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 TEST RAG v7 - Strict Local Mode")
    print("   Cải thiện độ chính xác với local Llama")
    print("=" * 60)
    
    # Force re-index?
    if args.force and os.path.exists(RAG_STORAGE):
        print(f"\n⚠️ Đang xóa database cũ: {RAG_STORAGE}")
        import shutil
        shutil.rmtree(RAG_STORAGE)
        print("   ✅ Đã xóa")
    
    os.makedirs(RAG_STORAGE, exist_ok=True)
    
    # Load models
    print("\n🔄 Loading models...")
    local_llm = create_local_llm()
    print("   ✅ Llama 3.1 8B loaded")
    embedding = create_embedding()
    print("   ✅ Embedding loaded")
    
    # Initialize LightRAG
    print("\n🔧 Initializing LightRAG...")
    rag = LightRAG(
        working_dir=RAG_STORAGE,
        llm_model_func=local_llm,
        embedding_func=embedding,
    )
    await rag.initialize_storages()
    print("✅ LightRAG ready!")
    
    # Auto-index
    print("\n" + "=" * 60)
    print("📁 KIỂM TRA FILE MỚI")
    print("=" * 60)
    
    tracker = IndexTracker(INDEX_TRACKER_FILE)
    
    if os.path.exists(COURSES_FOLDER):
        all_files = [f for f in os.listdir(COURSES_FOLDER) 
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        
        new_files = [f for f in all_files 
                     if tracker.needs_indexing(os.path.join(COURSES_FOLDER, f))]
        
        if new_files:
            print(f"\n🆕 Phát hiện {len(new_files)} file cần index:")
            for filename in new_files:
                file_path = os.path.join(COURSES_FOLDER, filename)
                try:
                    start = time.time()
                    text = read_file(file_path)
                    print(f"   📄 {filename} ({len(text)} chars)...")
                    await rag.ainsert(text)
                    tracker.mark_indexed(file_path)
                    print(f"   ✅ {filename} ({time.time()-start:.1f}s)")
                except Exception as e:
                    print(f"   ❌ {filename}: {e}")
        else:
            print(f"✅ Không có file mới. Database: {tracker.get_indexed_count()} files")
    
    # Interactive query
    print("\n" + "=" * 60)
    print("💬 CHẾ ĐỘ HỎI ĐÁP (Strict Local)")
    print("=" * 60)
    print("Gõ câu hỏi. 'exit' để thoát.")
    print("'mode:naive' để dùng tìm kiếm trực tiếp (khuyên dùng)")
    print("'mode:hybrid/local/global' để dùng RAG graph")
    print("-" * 60)
    
    current_mode = "naive"  # Default naive cho local LLM
    
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
            
            try:
                if current_mode == "naive":
                    # Tìm kiếm trực tiếp + answer với context
                    chunks = await direct_search(user_input, RAG_STORAGE, top_k=5)
                    result = await answer_with_context(user_input, chunks, local_llm)
                else:
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
