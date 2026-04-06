"""
Qwen2.5-14B RAG System
Kết hợp LightRAG với Qwen2.5-14B-Instruct
- Chặn token ngoại ngữ (Trung, Nga, Nhật, Hàn...)
- Auto-index files từ ./courses
- Query với nhiều mode: hybrid, local, global, naive

Chạy: uv run Qwen2.5_14B_RAG.py
"""

import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

import asyncio
import json
import hashlib
import os
import time
from datetime import datetime
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# ======================== CONFIG ========================
COURSES_FOLDER = "./courses"
RAG_STORAGE = "./rag_storage_qwen14b"
INDEX_TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")

SUPPORTED_EXTENSIONS = [".txt", ".md", ".csv", ".pdf"]

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512

# LLM
LLM_MAX_NEW_TOKENS = 512
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


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


# ======================== QWEN RAG CLASS ========================
class QwenRAG:
    def __init__(self):
        print("=" * 60)
        print("🚀 QWEN 2.5-14B RAG SYSTEM")
        print("=" * 60)
        
        # 1. Load Qwen model
        self._load_qwen_model()
        
        # 2. Load Embedding model
        self._load_embedding_model()
        
        # 3. Initialize LightRAG
        self._init_lightrag()
        
        # System prompt cho RAG
        self.system_prompt = """Bạn là trợ lý AI chuyên giáo dục.
Sử dụng thông tin được cung cấp để trả lời câu hỏi.
CHỈ trả lời bằng tiếng Việt.
Nếu không có thông tin trong tài liệu, hãy nói thẳng.
Trả lời ngắn gọn, chính xác."""

    def _load_qwen_model(self):
        print(f"\n🔄 Loading Qwen model: {MODEL_NAME}...")
        start = time.time()
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        
        print(f"   ✅ Model loaded ({time.time()-start:.1f}s)")
        
        # Tạo bad_words_ids để chặn token ngoại ngữ
        print("   🔧 Tạo danh sách chặn token ngoại ngữ...")
        self.bad_words_ids = self._get_non_vietnamese_bad_words()
        print(f"   ✅ Đã chặn {len(self.bad_words_ids)} token ngoại ngữ!")
    
    def _get_non_vietnamese_bad_words(self):
        """Chặn token KHÔNG PHẢI tiếng Việt/Latin"""
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
    
    def _load_embedding_model(self):
        print(f"\n🔄 Loading Embedding model: {EMBEDDING_MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("   ✅ Embedding loaded!")
    
    def _init_lightrag(self):
        print("\n🔧 Initializing LightRAG...")
        os.makedirs(RAG_STORAGE, exist_ok=True)
        
        # Tạo async LLM function cho LightRAG
        async def qwen_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return self._generate_response(prompt, system_prompt)
        
        # Tạo async embedding function
        async def embedding_func(texts):
            return self.embedder.encode(texts)
        
        embedding_wrapper = EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKENS,
            func=embedding_func
        )
        
        self.rag = LightRAG(
            working_dir=RAG_STORAGE,
            llm_model_func=qwen_llm_func,
            embedding_func=embedding_wrapper,
        )
        print("   ✅ LightRAG initialized!")
    
    def _generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response từ Qwen model (sync)"""
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
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
                max_new_tokens=LLM_MAX_NEW_TOKENS,
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
    
    async def initialize(self):
        """Khởi tạo async cho LightRAG"""
        await self.rag.initialize_storages()
        print("✅ RAG storages ready!")
    
    async def auto_index_new_files(self):
        """Tự động phát hiện và index file mới"""
        tracker = IndexTracker(INDEX_TRACKER_FILE)
        
        if not os.path.exists(COURSES_FOLDER):
            print(f"⚠️ Folder {COURSES_FOLDER} không tồn tại")
            return 0
        
        all_files = []
        for f in os.listdir(COURSES_FOLDER):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_files.append(f)
        
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
        
        print("\n📥 Đang index...")
        indexed = 0
        
        for i, filename in enumerate(new_files, 1):
            file_path = os.path.join(COURSES_FOLDER, filename)
            
            try:
                start = time.time()
                text = read_file(file_path)
                await self.rag.ainsert(text)
                tracker.mark_indexed(file_path)
                elapsed = time.time() - start
                print(f"   [{i}/{len(new_files)}] ✅ {filename} ({elapsed:.1f}s)")
                indexed += 1
            except Exception as e:
                print(f"   [{i}/{len(new_files)}] ❌ {filename}: {e}")
        
        print(f"\n📊 Đã index: {indexed}/{len(new_files)} files")
        return indexed
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query RAG với question"""
        try:
            result = await self.rag.aquery(question, param=QueryParam(mode=mode))
            return result
        except Exception as e:
            return f"❌ Lỗi: {e}"
    
    async def interactive_mode(self):
        """Chế độ hỏi đáp tương tác"""
        print("\n" + "=" * 60)
        print("💬 CHẾ ĐỘ HỎI ĐÁP")
        print("=" * 60)
        print("Gõ câu hỏi và Enter. 'exit' để thoát.")
        print("'mode:hybrid/local/global/naive' để đổi mode")
        print("'clear' để xóa màn hình")
        print("-" * 60)
        
        current_mode = "hybrid"
        
        while True:
            try:
                user_input = input(f"\n🧑 [{current_mode}] Bạn: ").strip()
                
                if user_input.lower() in ["exit", "quit", "q", "thoát"]:
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
                
                if user_input.lower() == "clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                print("🤖 Đang xử lý...")
                start = time.time()
                result = await self.query(user_input, mode=current_mode)
                elapsed = time.time() - start
                print(f"\n🤖 AI ({elapsed:.1f}s):\n{result}")
                
            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break


# ======================== MAIN ========================
async def main():
    # 1. Khởi tạo hệ thống
    qwen_rag = QwenRAG()
    await qwen_rag.initialize()
    
    # 2. Auto-index new files
    print("\n" + "=" * 60)
    print("📁 KIỂM TRA FILE MỚI")
    print("=" * 60)
    await qwen_rag.auto_index_new_files()
    
    # 3. Test queries
    print("\n" + "=" * 60)
    print("🧪 TEST QUERIES")
    print("=" * 60)
    
    test_questions = [
        "RAG là gì?",
        "Machine Learning có những loại nào?",
    ]
    
    for q in test_questions:
        print(f"\n❓ {q}")
        start = time.time()
        answer = await qwen_rag.query(q, mode="hybrid")
        elapsed = time.time() - start
        # Truncate long answers
        if len(answer) > 300:
            print(f"📝 ({elapsed:.1f}s): {answer[:300]}...")
        else:
            print(f"📝 ({elapsed:.1f}s): {answer}")
    
    # 4. Interactive mode
    print("\n" + "=" * 60)
    print("💡 Vào chế độ hỏi đáp? (y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        await qwen_rag.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
