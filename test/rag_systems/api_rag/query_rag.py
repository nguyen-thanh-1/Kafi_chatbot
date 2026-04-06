"""
Query RAG - Script hỏi đáp nhanh (load từ database đã index)
Không cần parse lại tài liệu, chỉ load và query

Chạy: uv run query_rag.py
Hoặc: uv run query_rag.py "câu hỏi của bạn"
"""

import asyncio
import argparse
import os

# Import RAGAnything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# Import config
from rag_config import (
    RAG_STORAGE, EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    RAG_CONFIG
)


# ======================== MODELS SETUP (nhẹ hơn indexer) ========================
def setup_models():
    """Load và setup các models (embedding, LLM) - không cần MinerU parser"""
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


# ======================== RAG QUERY ========================
class RAGQuerier:
    """Lightweight RAG querier - chỉ load database và query"""
    
    def __init__(self):
        self.rag = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize RAG từ existing storage"""
        if self.initialized:
            return
        
        # Check if database exists
        if not os.path.exists(RAG_STORAGE):
            raise FileNotFoundError(
                f"❌ Database không tồn tại: {RAG_STORAGE}\n"
                f"   Chạy 'uv run index_docs.py' để index tài liệu trước!"
            )
        
        # Check for indexed data
        vdb_file = os.path.join(RAG_STORAGE, "vdb_chunks.json")
        if not os.path.exists(vdb_file):
            raise FileNotFoundError(
                f"❌ Database rỗng, chưa có dữ liệu index.\n"
                f"   Chạy 'uv run index_docs.py' để index tài liệu!"
            )
        
        # Setup models
        embedding, llm_func = setup_models()
        
        # Initialize RAG (sẽ tự động load existing data)
        print("\n🔧 Loading RAG database...")
        config = RAGAnythingConfig(**RAG_CONFIG)
        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            embedding_func=embedding,
        )
        
        print(f"✅ RAG loaded từ {RAG_STORAGE}")
        self.initialized = True
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """Query RAG với câu hỏi"""
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.rag.aquery(question, mode=mode)
            return result
        except Exception as e:
            return f"❌ Lỗi query: {str(e)}"


# ======================== INTERACTIVE MODE ========================
async def interactive_mode(querier: RAGQuerier):
    """Chế độ hỏi đáp tương tác"""
    print("\n" + "=" * 60)
    print("💬 CHẾ ĐỘ HỎI ĐÁP TƯƠNG TÁC")
    print("=" * 60)
    print("Nhập câu hỏi và nhấn Enter. Gõ 'exit' để thoát.")
    print("Gõ 'mode:hybrid', 'mode:local', 'mode:global' để đổi chế độ query.")
    print("-" * 60)
    
    current_mode = "hybrid"
    
    while True:
        try:
            user_input = input(f"\n🧑 [{current_mode}] Bạn: ").strip()
            
            # Check exit
            if user_input.lower() in ["exit", "quit", "q", "thoát"]:
                print("👋 Tạm biệt!")
                break
            
            # Check empty
            if not user_input:
                continue
            
            # Check mode change
            if user_input.startswith("mode:"):
                new_mode = user_input.split(":")[1].strip()
                if new_mode in ["hybrid", "local", "global", "naive"]:
                    current_mode = new_mode
                    print(f"✅ Đổi sang chế độ: {current_mode}")
                else:
                    print("❌ Mode không hợp lệ. Các mode: hybrid, local, global, naive")
                continue
            
            # Query
            print("🤖 Đang xử lý...")
            result = await querier.query(user_input, mode=current_mode)
            print(f"\n🤖 AI:\n{result}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except EOFError:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")


# ======================== SINGLE QUERY MODE ========================
async def single_query(question: str, mode: str = "hybrid"):
    """Chạy một query duy nhất và thoát"""
    print("=" * 60)
    print("🔍 RAG QUERY")
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
    """Main async entry point"""
    if args.question:
        # Single query mode
        await single_query(args.question, mode=args.mode)
    else:
        # Interactive mode
        print("=" * 60)
        print("🚀 RAG QUERIER")
        print("=" * 60)
        
        querier = RAGQuerier()
        await querier.initialize()
        await interactive_mode(querier)


def main():
    parser = argparse.ArgumentParser(description="Query RAG database")
    parser.add_argument(
        'question',
        nargs='?',
        default=None,
        help='Câu hỏi để query (nếu không có sẽ vào chế độ interactive)'
    )
    parser.add_argument(
        '--mode', '-m',
        default='hybrid',
        choices=['hybrid', 'local', 'global', 'naive'],
        help='Chế độ query (default: hybrid)'
    )
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
