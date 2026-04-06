"""
Query RAG từ database được index bởi Gemini
Vẫn dùng Local Llama 3.1 để query (bảo mật, miễn phí)

Chạy: uv run query_rag_gemini.py
Hoặc: uv run query_rag_gemini.py "câu hỏi"
"""

import asyncio
import argparse
import os

# Import RAGAnything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# ======================== PATHS (khớp với index_docs_gemini.py) ========================
RAG_STORAGE = "./rag_storage_gemini"

# Embedding config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 256

# LLM settings (local)
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.1


# ======================== MODELS SETUP ========================
def setup_models():
    """Load local models cho query"""
    print("🔄 Loading models...")
    
    # Import Llama local
    try:
        from Llama_3_1_8B_Instruct_v2 import generate_response
        print("   ✅ Llama 3.1 8B loaded (local)")
    except ImportError as e:
        print(f"   ❌ Không thể load Llama: {e}")
        raise
    
    # Embedding
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("   ❌ Chưa cài sentence-transformers")
        raise
    
    print(f"   Loading embedding: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("   ✅ Embedding loaded")
    
    async def embedding_func(texts):
        return embedder.encode(texts)
    
    embedding = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embedding_func
    )
    
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
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


# ======================== RAG QUERIER ========================
class RAGQuerier:
    """Query từ database Gemini"""
    
    def __init__(self):
        self.rag = None
        self.initialized = False
    
    async def initialize(self):
        if self.initialized:
            return
        
        # Check database
        if not os.path.exists(RAG_STORAGE):
            raise FileNotFoundError(
                f"❌ Database không tồn tại: {RAG_STORAGE}\n"
                f"   Chạy 'uv run index_docs_gemini.py' để index trước!"
            )
        
        vdb_file = os.path.join(RAG_STORAGE, "vdb_chunks.json")
        if not os.path.exists(vdb_file):
            raise FileNotFoundError(
                f"❌ Database rỗng.\n"
                f"   Chạy 'uv run index_docs_gemini.py' để index!"
            )
        
        # Setup
        embedding, llm_func = setup_models()
        
        print("\n🔧 Loading RAG database (Gemini-indexed)...")
        config = RAGAnythingConfig(
            working_dir=RAG_STORAGE,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=False,
        )
        
        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            embedding_func=embedding,
        )
        
        print(f"✅ RAG loaded từ {RAG_STORAGE}")
        self.initialized = True
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.rag.aquery(question, mode=mode)
            return result
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"


# ======================== INTERACTIVE MODE ========================
async def interactive_mode(querier: RAGQuerier):
    print("\n" + "=" * 60)
    print("💬 CHẾ ĐỘ HỎI ĐÁP (Gemini Database + Local LLM)")
    print("=" * 60)
    print("Gõ câu hỏi và nhấn Enter. 'exit' để thoát.")
    print("Gõ 'mode:hybrid/local/global' để đổi mode.")
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
                    print(f"✅ Đổi mode: {current_mode}")
                else:
                    print("❌ Mode không hợp lệ")
                continue
            
            print("🤖 Đang xử lý...")
            result = await querier.query(user_input, mode=current_mode)
            print(f"\n🤖 AI:\n{result}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")


# ======================== SINGLE QUERY ========================
async def single_query(question: str, mode: str = "hybrid"):
    print("=" * 60)
    print("🔍 RAG QUERY (Gemini Database)")
    print("=" * 60)
    
    querier = RAGQuerier()
    await querier.initialize()
    
    print(f"\n❓ Câu hỏi: {question}")
    print(f"📌 Mode: {mode}")
    print("-" * 60)
    
    result = await querier.query(question, mode=mode)
    print(f"\n📝 Kết quả:\n{result}")


# ======================== MAIN ========================
async def main_async(args):
    if args.question:
        await single_query(args.question, mode=args.mode)
    else:
        print("=" * 60)
        print("🚀 RAG QUERIER (Gemini Database + Local LLM)")
        print("=" * 60)
        
        querier = RAGQuerier()
        await querier.initialize()
        await interactive_mode(querier)


def main():
    parser = argparse.ArgumentParser(description="Query Gemini-indexed RAG")
    parser.add_argument('question', nargs='?', default=None)
    parser.add_argument('--mode', '-m', default='hybrid',
                       choices=['hybrid', 'local', 'global', 'naive'])
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
