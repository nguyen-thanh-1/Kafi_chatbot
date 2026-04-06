"""
Test RAG mẫu sử dụng RAGAnything với Local LLM (Llama 3.1)
Thư viện: https://github.com/HKUDS/RAG-Anything
Cài đặt: pip install raganything sentence-transformers
"""

import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import torch

# Import model local
try:
    from Llama_3_1_8B_Instruct_v2 import generate_response
except ImportError:
    print("Lỗi: Không tìm thấy file Llama_3_1_8B_Instruct_v2.py hoặc không thể import model.")
    exit(1)

# Import sentence_transformers cho embedding
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Lỗi: Chưa cài đặt sentence-transformers. Vui lòng chạy: pip install sentence-transformers")
    exit(1)


# ======================== CẤU HÌNH LOCAL EMBEDDING ========================
# Load model embedding (nhẹ, chạy CPU/GPU đều ổn)
embedding_model_name = "all-MiniLM-L6-v2"
print(f"Loading embedding model: {embedding_model_name}...")
embedder = SentenceTransformer(embedding_model_name)
print("Embedding model loaded!")

async def local_embedding_func(texts):
    """Hàm tạo embedding sử dụng sentence-transformers (async wrapper)"""
    # LightRAG expects numpy array with .size attribute, NOT list
    return embedder.encode(texts)

embedding_func = EmbeddingFunc(
    embedding_dim=384, # all-MiniLM-L6-v2 có dim là 384
    max_token_size=256, # 512, nhưng để an toàn 256
    func=local_embedding_func
)


# ======================== CẤU HÌNH LOCAL LLM ========================

async def local_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Bridge function gọi tới Llama 3.1 local (async wrapper)"""
    print(f"🤖 Calling Local LLM with prompt len: {len(prompt)}")
    
    # RAGAnything/LightRAG có thể truyền history_messages phức tạp,
    # nhưng ở đây ta đơn giản hóa truyền vào hàm generate_response
    
    # Chuẩn bị history format cho hàm generate_response
    # Hàm generate_response mong đợi history là list dict [{"role":..., "content":...}]
    
    # Nếu có history_messages từ RAG, ta dùng nó
    chat_history = history_messages if history_messages else []
    
    # Gọi hàm generate từ file script của user
    response = generate_response(
        user_input=prompt,
        history=chat_history,
        system_prompt=system_prompt,
        max_new_tokens=1024, # Tăng token cho câu trả lời dài hơn
        temperature=0.1 # Giảm temperature để trả lời chính xác hơn cho RAG
    )
    
    return response


async def main():
    """Hàm chính để chạy RAG với folder documents"""
    
    # ======================== CẤU HÌNH ========================
    COURSES_FOLDER = "./courses"  # Folder chứa các tài liệu nguồn
    OUTPUT_DIR = "./output_courses"  # Folder output cho parsed documents
    RAG_STORAGE = "./rag_storage_courses"  # Folder lưu RAG database
    
    # Các định dạng file được hỗ trợ
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pptx", ".ppt", ".md"]
    
    # ======================== CẤU HÌNH RAG ========================
    config = RAGAnythingConfig(
        working_dir=RAG_STORAGE,
        parser="mineru",  # Parser mạnh mẽ, hỗ trợ nhiều format
        parse_method="auto",  # Tự động chọn phương thức parse phù hợp
        enable_image_processing=False,  # Tắt để tăng tốc
        enable_table_processing=True,  # Bật xử lý bảng (hữu ích cho Excel/CSV)
        enable_equation_processing=False,
    )
    
    # ======================== KHỞI TẠO RAG ========================
    rag = RAGAnything(
        config=config,
        llm_model_func=local_llm_func,
        embedding_func=embedding_func,
    )
    
    print("✅ RAG Initialized with Local LLM & Embeddings")
    print(f"📁 Courses folder: {COURSES_FOLDER}")
    print(f"📂 Output folder: {OUTPUT_DIR}")
    print(f"💾 RAG storage: {RAG_STORAGE}")
    
    # ======================== XỬ LÝ TẤT CẢ TÀI LIỆU TRONG FOLDER ========================
    import os
    
    # Kiểm tra folder tồn tại
    if not os.path.exists(COURSES_FOLDER):
        print(f"❌ Folder không tồn tại: {COURSES_FOLDER}")
        return
    
    # Liệt kê các file được hỗ trợ
    files = []
    for f in os.listdir(COURSES_FOLDER):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(f)
    
    if not files:
        print(f"⚠️ Không tìm thấy file nào được hỗ trợ trong {COURSES_FOLDER}")
        print(f"   Các định dạng hỗ trợ: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"\n📋 Tìm thấy {len(files)} file(s) để xử lý:")
    for i, f in enumerate(files, 1):
        file_path = os.path.join(COURSES_FOLDER, f)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   {i}. {f} ({size_mb:.2f} MB)")
    
    # Xử lý từng file
    print(f"\n🔄 Bắt đầu xử lý documents...")
    
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(COURSES_FOLDER, filename)
        print(f"\n[{i}/{len(files)}] Processing: {filename}")
        
        try:
            await rag.process_document_complete(
                file_path=file_path,
                output_dir=OUTPUT_DIR,
                parse_method="auto"  # Tự động detect phương thức phù hợp
            )
            print(f"   ✅ Đã xử lý: {filename}")
        except Exception as e:
            print(f"   ❌ Lỗi khi xử lý {filename}: {str(e)}")
            continue
    
    print(f"\n✅ Hoàn tất xử lý {len(files)} tài liệu!")
    
    # ======================== DEMO TRUY VẤN ========================
    print("\n" + "="*60)
    print("🔍 DEMO TRUY VẤN")
    print("="*60)
    
    # Các câu hỏi mẫu
    queries = [
        "Tóm tắt nội dung chính của các tài liệu",
        "Event-Driven Design là gì?",
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        try:
            result = await rag.aquery(query, mode="hybrid")
            print(f"📝 Kết quả:\n{result}")
        except Exception as e:
            print(f"❌ Lỗi query: {str(e)}")
        print("-" * 40)
    
    # ======================== CHẾ ĐỘ HỎI ĐÁP TƯƠNG TÁC ========================
    print("\n" + "="*60)
    print("💬 CHẾ ĐỘ HỎI ĐÁP (gõ 'exit' để thoát)")
    print("="*60)
    
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
            print("\n� Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
