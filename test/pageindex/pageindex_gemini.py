"""
PageIndex Multi-Format RAG với Gemini API
Wrapper để dễ dàng set API key
"""

import os
from pageindex_multiformat import MultiFormatRAG

# ═══════════════════════════════════════════════════════════
# CONFIG - THAY ĐỔI API KEY Ở ĐÂY
# ═══════════════════════════════════════════════════════════
GEMINI_API_KEY = "[ENCRYPTION_KEY]"  # ← API key của bạn
MODEL_NAME = "gemini-2.5-flash"   # Model name
DOCUMENTS_DIR = "./courses"

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🚀 PageIndex Multi-Format RAG System (Gemini API)")
    print("=" * 70)
    print("\n📌 Hỗ trợ định dạng:")
    print("  ✅ TXT - Text files")
    print("  ✅ PDF - PDF documents")
    print("  ✅ DOCX - Word documents")
    print("  ✅ MD - Markdown files")
    print(f"\n🤖 LLM: {MODEL_NAME}")
    print(f"🔑 API Key: {GEMINI_API_KEY[:20]}...")
    print("=" * 70)
    
    try:
        # Khởi tạo RAG với API key
        rag = MultiFormatRAG(
            documents_dir=DOCUMENTS_DIR,
            api_key=GEMINI_API_KEY,
            model_name=MODEL_NAME
        )
    except Exception as e:
        print(f"\n❌ Lỗi khởi tạo: {e}")
        return
    
    # Hiển thị thống kê
    stats = rag.get_statistics()
    print(f"\n📊 Thống kê:")
    print(f"  • Tổng tài liệu: {stats['total_documents']}")
    print(f"  • Tổng sections: {stats['total_sections']}")
    print(f"  • Theo loại:")
    for file_type, count in stats['by_type'].items():
        print(f"    - {file_type.upper()}: {count} files")
    
    print("\n✅ Hệ thống sẵn sàng!")
    print("\n📝 Lệnh: rebuild | stats | exit")
    print("=" * 70)
    
    # Interactive loop
    while True:
        print("\n")
        user_input = input("💬 Câu hỏi: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Tạm biệt!")
            break
        
        if user_input.lower() == "rebuild":
            rag.rebuild_index()
            stats = rag.get_statistics()
            print(f"✅ Rebuild xong! {stats['total_documents']} docs, {stats['total_sections']} sections")
            continue
        
        if user_input.lower() == "stats":
            stats = rag.get_statistics()
            print(f"\n📊 Thống kê chi tiết:")
            for doc in stats['documents']:
                print(f"  • {doc['name']} ({doc['type'].upper()}): {doc['sections']} sections")
            continue
        
        print("\n🤖 Đang xử lý...")
        print("=" * 70)
        
        try:
            response, sources = rag.query(user_input)
            print(f"\n📝 Trả lời:\n{response}")
            
            if sources:
                print(f"\n📚 Nguồn:")
                for idx, source in enumerate(sources, 1):
                    print(f"  {idx}. {source}")
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
        
        print("=" * 70)


if __name__ == "__main__":
    main()
