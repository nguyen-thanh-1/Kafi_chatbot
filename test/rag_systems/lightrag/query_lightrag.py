"""
Query LightRAG - Hỏi đáp từ database LightRAG
Dùng Local Llama 3.1 để query (bảo mật, miễn phí)

Chạy: uv run query_lightrag.py
Hoặc: uv run query_lightrag.py "câu hỏi"
"""

import asyncio
import argparse
import os
import numpy as np

# Import LightRAG
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# ======================== CONFIG ========================
RAG_STORAGE = "./rag_storage_lightrag"

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512

# LLM
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.1


# ======================== MODELS ========================
def setup_models():
    """Load models cho query"""
    print("🔄 Loading models...")
    
    # Local Llama
    try:
        from Llama_3_1_8B_Instruct_v2 import generate_response
        print("   ✅ Llama 3.1 8B loaded")
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
    
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return embedder.encode(texts)
    
    embedding = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=EMBEDDING_MAX_TOKENS,
        func=embedding_func
    )
    
    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
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


# ======================== QUERIER ========================
class LightRAGQuerier:
    def __init__(self):
        self.rag = None
        self.initialized = False
    
    async def initialize(self):
        if self.initialized:
            return
        
        if not os.path.exists(RAG_STORAGE):
            raise FileNotFoundError(
                f"❌ Database không tồn tại: {RAG_STORAGE}\n"
                f"   Chạy 'uv run index_lightrag.py' trước!"
            )
        
        # Check for data
        graph_file = os.path.join(RAG_STORAGE, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graph_file):
            raise FileNotFoundError(
                f"❌ Database rỗng.\n"
                f"   Chạy 'uv run index_lightrag.py' để index!"
            )
        
        embedding, llm_func = setup_models()
        
        print("\n🔧 Loading LightRAG database...")
        self.rag = LightRAG(
            working_dir=RAG_STORAGE,
            llm_model_func=llm_func,
            embedding_func=embedding,
        )
        
        # QUAN TRỌNG: Phải initialize storages
        await self.rag.initialize_storages()
        
        print(f"✅ LightRAG loaded từ {RAG_STORAGE}")
        self.initialized = True
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.rag.aquery(
                question,
                param=QueryParam(mode=mode)
            )
            return result
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"


# ======================== INTERACTIVE ========================
async def interactive_mode(querier: LightRAGQuerier):
    print("\n" + "=" * 60)
    print("💬 LightRAG Q&A (Local LLM)")
    print("=" * 60)
    print("Gõ câu hỏi. 'exit' để thoát.")
    print("Gõ 'mode:hybrid/local/global/naive' để đổi mode.")
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
                else:
                    print("❌ Modes: hybrid, local, global, naive")
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
    print("🔍 LightRAG QUERY")
    print("=" * 60)
    
    querier = LightRAGQuerier()
    await querier.initialize()
    
    print(f"\n❓ {question}")
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
        print("🚀 LightRAG QUERIER")
        print("=" * 60)
        
        querier = LightRAGQuerier()
        await querier.initialize()
        await interactive_mode(querier)


def main():
    parser = argparse.ArgumentParser(description="Query LightRAG")
    parser.add_argument('question', nargs='?', default=None)
    parser.add_argument('--mode', '-m', default='hybrid',
                       choices=['hybrid', 'local', 'global', 'naive'])
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
