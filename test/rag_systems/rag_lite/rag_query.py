#!/usr/bin/env python3
"""
RAG Lite - Query Only Script
Fast querying with FlashRank reranker
"""

import argparse
from rag_lite import (
    RAGLite,
    get_embedder,
    get_reranker,
    get_llm,
    EMBEDDING_MODEL,
    CHUNK_SIZE
)


def main():
    parser = argparse.ArgumentParser(description="RAG Lite - Query Documents")
    parser.add_argument('--query', '-q', type=str, help='Single query mode')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG LITE - QUERY ONLY")
    print("═" * 60)
    print(f"   📊 Embedding: {EMBEDDING_MODEL}")
    print(f"   🎯 Reranker:  FlashRank (ONNX)")
    print(f"   🤖 LLM:       Llama 3.1 8B (GPU)")
    print(f"   ⚡ Chunking:  Recursive ({CHUNK_SIZE} chars)")
    print("═" * 60)
    
    # Load LLM first (GPU priority)
    print("\n🔄 Loading LLM (GPU priority)...")
    get_llm()
    
    # Load index
    print("\n🔄 Loading index from disk...")
    rag = RAGLite()
    if not rag.load():
        print("\n❌ Lỗi: Index not found!")
        print("\n💡 Hãy chạy rag_index.py trước để tạo index!")
        return
    
    print(f"   ✅ Loaded {len(rag.vector_store.chunks)} chunks")
    
    # Load other models
    print("\n🔄 Loading embedding & reranker...")
    get_embedder()
    get_reranker()
    
    # Stats
    print("\n" + "═" * 60)
    print("📊 DATABASE STATS")
    print("═" * 60)
    print(f"   Total files: {rag.tracker.get_indexed_count()}")
    print(f"   Total chunks: {rag.tracker.get_total_chunks()}")
    print("═" * 60)
    
    # Single query mode
    if args.query:
        print(f"\n❓ {args.query}")
        print("\n🤖 Đang xử lý...")
        answer = rag.query(args.query)
        print(f"\n📝 Trả lời:\n{answer}")
        return
    
    # Interactive mode
    print("\n" + "═" * 60)
    print("💬 INTERACTIVE MODE")
    print("═" * 60)
    print("Gõ câu hỏi. 'exit' để thoát, 'stats' để xem thống kê.")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🧑 Bạn: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break
            
            if question.lower() == "stats":
                print(f"\n📊 Files: {rag.tracker.get_indexed_count()}")
                print(f"📊 Chunks: {rag.tracker.get_total_chunks()}")
                continue
            
            if not question:
                continue
            
            print("\n🤖 Đang xử lý...")
            answer = rag.query(question)
            print(f"\n📝 Trả lời:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break


if __name__ == "__main__":
    main()
