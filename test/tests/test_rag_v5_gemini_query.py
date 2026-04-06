"""
Test RAG v5 - LightRAG với Gemini Query
- Index: Dùng database đã có (rag_storage_v4)
- Query: Dùng Gemini API (chính xác hơn local Llama)

Chạy: uv run test_rag_v5_gemini_query.py

Yêu cầu: GEMINI_API_KEY trong file .env
"""

import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Import LightRAG
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np

# ======================== CONFIG ========================
RAG_STORAGE = "./rag_storage_v4"  # Dùng database đã index

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 512


# ======================== GEMINI LLM ========================
def create_gemini_llm():
    """Tạo Gemini LLM function cho query"""
    import google.generativeai as genai
    
    if not GEMINI_API_KEY:
        raise ValueError("❌ Thiếu GEMINI_API_KEY trong .env")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    async def gemini_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
        
        full_prompt += prompt
        
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            return f"❌ Gemini error: {e}"
    
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
    print("=" * 60)
    print("🚀 TEST RAG v5 - Gemini Query")
    print("   Database: rag_storage_v4 (đã index)")
    print("   Query LLM: Gemini API (chính xác hơn)")
    print("=" * 60)
    
    if not GEMINI_API_KEY:
        print("\n❌ Thiếu GEMINI_API_KEY!")
        print("   Thêm vào file .env: GEMINI_API_KEY=your_key")
        return
    
    print(f"✅ Gemini API Key: {GEMINI_API_KEY[:10]}...")
    
    # Check database
    if not os.path.exists(RAG_STORAGE):
        print(f"\n❌ Database không tồn tại: {RAG_STORAGE}")
        print("   Chạy test_rag_v4_lightrag.py trước để index!")
        return
    
    # Load models
    print("\n🔄 Loading models...")
    gemini_llm = create_gemini_llm()
    print("   ✅ Gemini LLM ready")
    embedding = create_embedding()
    print("   ✅ Embedding loaded")
    
    # Initialize LightRAG
    print("\n🔧 Loading LightRAG...")
    rag = LightRAG(
        working_dir=RAG_STORAGE,
        llm_model_func=gemini_llm,
        embedding_func=embedding,
    )
    await rag.initialize_storages()
    print("✅ LightRAG ready!")
    
    # Interactive query
    print("\n" + "=" * 60)
    print("💬 CHẾ ĐỘ HỎI ĐÁP (Gemini Query)")
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
            
            result = await rag.aquery(user_input, param=QueryParam(mode=current_mode))
            
            elapsed = time.time() - start
            print(f"\n🤖 AI ({elapsed:.1f}s):\n{result}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    asyncio.run(main())
