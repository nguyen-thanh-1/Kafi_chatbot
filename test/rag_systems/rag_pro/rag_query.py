"""
RAG Pro V2 - QUERY ONLY
═══════════════════════════════════════════════════════════
Chỉ dùng để query, KHÔNG index.
Load index từ disk, LLM chạy trên GPU.

USAGE:
  uv run rag_query.py                    # Interactive mode
  uv run rag_query.py --query "câu hỏi" # Single query
  
PERFORMANCE:
  - LLM GPU: Nhanh
  - VRAM: ~12GB (LLM)
  - Query time: ~7.5s
═══════════════════════════════════════════════════════════
"""

import argparse

# Import từ rag_pro_v2
from rag_pro_v2 import (
    RAG_STORAGE,
    EMBEDDING_CACHE_FILE,
    TRACKER_FILE,
    EmbeddingCache,
    IndexTracker,
    VectorStore,
    get_embedder,
    get_reranker,
    get_llm,
    embed_query,
    rerank,
    generate_answer,
    TOP_K_RETRIEVE,
    TOP_K_RERANK,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
)

import time


# ═══════════════════════════════════════════════════════════
# QUERY ENGINE
# ═══════════════════════════════════════════════════════════
class RAGQuery:
    """Chỉ dùng để query, không index"""
    
    def __init__(self):
        self.vector_store = VectorStore(RAG_STORAGE)
        self.tracker = IndexTracker(TRACKER_FILE)
        
        # Load index
        print("\n🔄 Loading index from disk...")
        if not self.vector_store.load():
            raise RuntimeError(
                "❌ Không tìm thấy index! Vui lòng chạy rag_index.py trước."
            )
        
        print(f"   ✅ Loaded {len(self.vector_store.chunks)} chunks")
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Query RAG pipeline"""
        start = time.time()
        
        # Step 1: Retrieve from FAISS
        if verbose:
            print(f"   🔍 Searching...")
        retrieved_chunks = self.vector_store.search(question, TOP_K_RETRIEVE)
        
        if not retrieved_chunks:
            return "Tôi không tìm thấy thông tin liên quan trong database."
        
        if verbose:
            print(f"   📄 Found {len(retrieved_chunks)} chunks")
        
        # Step 2: Rerank
        if verbose:
            print(f"   🎯 Reranking to top {TOP_K_RERANK}...")
        reranked = rerank(question, retrieved_chunks, TOP_K_RERANK)
        
        if verbose:
            print(f"   ✅ Selected {len(reranked)} best chunks")
        
        # Step 3: Generate answer
        if verbose:
            print(f"   🤖 Generating answer...")
        answer = generate_answer(question, reranked)
        
        elapsed = time.time() - start
        if verbose:
            print(f"   ⏱️ Total: {elapsed:.1f}s")
        
        return answer
    
    def get_stats(self):
        """Lấy thống kê"""
        return {
            'total_files': self.tracker.get_indexed_count(),
            'total_chunks': self.tracker.get_total_chunks(),
            'indexed_files': self.tracker.indexed_files
        }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RAG Pro V2 - Query Only")
    parser.add_argument('--query', '-q', type=str, help='Single query mode')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG PRO V2 - QUERY ONLY")
    print("═" * 60)
    print(f"   📊 Embedding: BAAI/bge-m3 (CPU)")
    print(f"   🎯 Reranker:  BAAI/bge-reranker-v2-m3 (CPU)")
    print(f"   🤖 LLM:       Llama 3.1 8B (GPU)")
    print(f"   ⚡ Chunking:  Semantic ({MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} words)")
    print("═" * 60)
    
    # CRITICAL: Load LLM FIRST to ensure it gets GPU
    print("\n🔄 Loading LLM (GPU priority)...")
    get_llm()
    
    # Initialize query engine (loads index)
    try:
        rag = RAGQuery()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("\n💡 Hãy chạy rag_index.py trước để tạo index!")
        return
    
    # Load embedding and reranker AFTER LLM (on CPU)
    print("\n🔄 Loading embedding & reranker (CPU)...")
    get_embedder()
    get_reranker()
    
    # Show stats
    stats = rag.get_stats()
    print("\n" + "═" * 60)
    print("📊 DATABASE STATS")
    print("═" * 60)
    print(f"   Total files: {stats['total_files']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print("═" * 60)
    
    # Single query mode
    if args.query:
        print("\n" + "═" * 60)
        print("🔍 QUERY")
        print("═" * 60)
        print(f"\n❓ {args.query}")
        answer = rag.query(args.query)
        print(f"\n🤖 Answer:\n{answer}")
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
                print("\n👋 Tạm biệt!")
                break
            
            if not question:
                continue
            
            if question.lower() == "stats":
                stats = rag.get_stats()
                print(f"\n📊 Thống kê:")
                print(f"   Files: {stats['total_files']}")
                print(f"   Chunks: {stats['total_chunks']}")
                continue
            
            print("\n🤖 Đang xử lý...")
            answer = rag.query(question)
            print(f"\n📝 Trả lời:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break


if __name__ == "__main__":
    main()
