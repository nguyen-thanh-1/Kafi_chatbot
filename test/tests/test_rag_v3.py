"""
RAG Demo - Script demo đầy đủ workflow Index + Query
Đây là bản all-in-one để test, trong production nên tách thành:
  - index_docs.py: Chạy offline để index
  - query_rag.py: Chạy online để query nhanh

Chạy: uv run test_rag_v3.py
"""

import asyncio
import os
from datetime import datetime

# Import RAGAnything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# Import shared config
from rag_config import (
    COURSES_FOLDER, OUTPUT_DIR, RAG_STORAGE,
    SUPPORTED_EXTENSIONS, EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    RAG_CONFIG, ensure_directories, get_supported_files, get_file_info
)


# ======================== MODELS SETUP ========================
print("=" * 60)
print("🚀 RAG DEMO - Local LLM + RAGAnything")
print("=" * 60)

# Import Llama model
try:
    from Llama_3_1_8B_Instruct_v2 import generate_response
    print("✅ Llama 3.1 8B loaded")
except ImportError:
    print("❌ Không tìm thấy file Llama_3_1_8B_Instruct_v2.py")
    exit(1)

# Import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Chưa cài đặt sentence-transformers")
    exit(1)

# Load embedding model
print(f"Loading embedding: {EMBEDDING_MODEL_NAME}...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ Embedding model loaded")


# ======================== ASYNC FUNCTIONS ========================
async def local_embedding_func(texts):
    """Async embedding function"""
    return embedder.encode(texts)

embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=EMBEDDING_MAX_TOKENS,
    func=local_embedding_func
)

async def local_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Async LLM function"""
    chat_history = history_messages if history_messages else []
    response = generate_response(
        user_input=prompt,
        history=chat_history,
        system_prompt=system_prompt,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE
    )
    return response


# ======================== MAIN ========================
async def main():
    """Main workflow: Index + Query"""
    
    # Ensure directories
    ensure_directories()
    
    print(f"\n📁 Cấu hình:")
    print(f"   - Tài liệu: {COURSES_FOLDER}")
    print(f"   - Output: {OUTPUT_DIR}")
    print(f"   - Database: {RAG_STORAGE}")
    
    # Initialize RAG
    print("\n🔧 Initializing RAGAnything...")
    config = RAGAnythingConfig(**RAG_CONFIG)
    rag = RAGAnything(
        config=config,
        llm_model_func=local_llm_func,
        embedding_func=embedding_func,
    )
    print("✅ RAGAnything initialized")
    
    # ======================== PHASE 1: INDEXING ========================
    print("\n" + "=" * 60)
    print("📚 PHASE 1: INDEXING DOCUMENTS")
    print("=" * 60)
    
    # Get files
    files = get_supported_files(COURSES_FOLDER)
    if not files:
        print(f"❌ Không tìm thấy file trong {COURSES_FOLDER}")
        print(f"   Hỗ trợ: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"📋 Tìm thấy {len(files)} file(s):")
    for i, f in enumerate(files, 1):
        info = get_file_info(COURSES_FOLDER, f)
        print(f"   {i}. {f} ({info['size_mb']:.2f} MB)")
    
    # Ask user confirmation
    print(f"\n⚠️  Bạn có muốn index {len(files)} file(s) không?")
    print("   Nhấn Enter để tiếp tục, hoặc 'skip' để bỏ qua indexing...")
    
    user_input = input("   > ").strip().lower()
    
    if user_input != "skip":
        # Index files
        for i, filename in enumerate(files, 1):
            file_path = os.path.join(COURSES_FOLDER, filename)
            print(f"\n[{i}/{len(files)}] Indexing: {filename}")
            
            try:
                start = datetime.now()
                await rag.process_document_complete(
                    file_path=file_path,
                    output_dir=OUTPUT_DIR,
                    parse_method="auto"
                )
                elapsed = (datetime.now() - start).total_seconds()
                print(f"   ✅ Done in {elapsed:.1f}s")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    else:
        print("⏭️  Skipped indexing")
    
    # ======================== PHASE 2: QUERYING ========================
    print("\n" + "=" * 60)
    print("🔍 PHASE 2: QUERYING")
    print("=" * 60)
    
    # Demo queries
    demo_queries = [
        "Tóm tắt nội dung chính của các tài liệu",
        "Event-Driven Design là gì và lợi ích của nó?",
    ]
    
    print("📝 Demo queries:")
    for query in demo_queries:
        print(f"\n❓ {query}")
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"� {result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # ======================== PHASE 3: INTERACTIVE ========================
    print("\n" + "=" * 60)
    print("💬 PHASE 3: INTERACTIVE Q&A")
    print("=" * 60)
    print("Gõ câu hỏi và nhấn Enter. Gõ 'exit' để thoát.")
    
    while True:
        try:
            user_query = input("\n🧑 Bạn: ").strip()
            
            if user_query.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break
            
            if not user_query:
                continue
            
            print("🤖 Đang xử lý...")
            result = await rag.aquery(user_query, mode="hybrid")
            print(f"🤖 AI: {result}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

