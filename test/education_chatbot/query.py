#!/usr/bin/env python3
"""
Query script for Education Chatbot (CLI mode)
"""

from rag_engine import RAGHybrid, get_embedder, get_reranker, get_llm


def main():
    print("═" * 60)
    print("💬 EDUCATION CHATBOT - QUERY")
    print("═" * 60)
    
    # Load models
    print("\n🔄 Loading models...")
    rag = RAGHybrid()
    rag.preload_lite()
    
    # Stats
    stats = rag.get_stats()
    print(f"\n📊 Index: {stats['files']} files, {stats['chunks']} chunks")
    
    if stats['chunks'] == 0:
        print("\n⚠️ No documents indexed!")
        print("   Run: python index.py")
        print("   Or use streamlit: streamlit run app.py")
        return
    
    # Interactive mode
    print("\n" + "═" * 60)
    print("💬 INTERACTIVE MODE")
    print("═" * 60)
    print("Gõ câu hỏi. 'exit' để thoát.")
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
            answer, mode = rag.query_with_mode(question, verbose=True)
            
            mode_text = {
                "rag_lite": "⚡ Hybrid Mode",
                "rag_pro": "📚 Strict Mode",
                "llm_only": "🤖 LLM Only"
            }.get(mode, mode)
            
            print(f"\n{mode_text}")
            print(f"\n📝 Trả lời:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break


if __name__ == "__main__":
    main()
