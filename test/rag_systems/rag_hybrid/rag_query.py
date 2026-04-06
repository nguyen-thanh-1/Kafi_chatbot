#!/usr/bin/env python3
"""
RAG Hybrid - Query Script
2-Stage RAG with Question Routing
"""

from rag_hybrid import RAGHybrid


def main():
    print("═" * 60)
    print("🚀 RAG HYBRID - 2-Stage Question Routing")
    print("═" * 60)
    print("   📊 Strategy: Question Router → rag_lite / rag_pro")
    print("   ⚡ Fast: RAG Lite + LLM General Knowledge")
    print("   📚 Deep: RAG Pro (Strict Document Only)")
    print("═" * 60)
    
    rag = RAGHybrid()
    
    print("\n🔄 Preloading RAG Lite + LLM...")
    rag.preload_lite()
    
    print("\n" + "═" * 60)
    print("💬 INTERACTIVE MODE")
    print("═" * 60)
    print("Gõ câu hỏi. 'exit' để thoát.")
    print("")
    print("💡 Routing Tips:")
    print("   → 'NLP là gì?' = Fast mode (hybrid, dùng LLM knowledge)")
    print("   → 'Theo tài liệu, NLP là gì?' = Deep mode (strict, chỉ tài liệu)")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🧑 Bạn: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break
            
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
