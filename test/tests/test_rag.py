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
    """Hàm chính để chạy RAG"""
    
    # ======================== CẤU HÌNH RAG ========================
    config = RAGAnythingConfig(
        working_dir="./rag_storage_local", # Thay đổi thư mục để không đè lên cái cũ
        parser="mineru", 
        parse_method="txt", # Dùng txt cho nhanh và đơn giản với demo
        enable_image_processing=False, # Tắt image processing vì model vision chưa setup local
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    
    # ======================== KHỞI TẠO RAG ========================
    rag = RAGAnything(
        config=config,
        llm_model_func=local_llm_func,
        # vision_model_func=vision_model_func, # Bỏ qua vision model cho demo text thuần
        embedding_func=embedding_func,
    )
    
    print("✅ RAG Initialized with Local LLM & Embeddings")
    
    # ======================== TẠO DỮ LIỆU MẪU ========================
    # Tạo một file txt mẫu để test
    sample_file = "sample_knowledge.txt"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("""
        RAGAnything là một framework RAG tất cả trong một.
        Nó hỗ trợ xử lý đa phương thức (hình ảnh, bảng biểu, công thức).
        Việc sử dụng Local LLM giúp bảo mật dữ liệu và tiết kiệm chi phí API.
        Llama 3.1 8B là một mô hình ngôn ngữ mạnh mẽ của Meta.
        """)
    
    # ======================== XỬ LÝ TÀI LIỆU ========================
    print(f"Processing {sample_file}...")
    await rag.process_document_complete(
        file_path=sample_file,
        output_dir="./output_local",
        parse_method="txt"
    )
    print(f"✅ Đã xử lý tài liệu")
    
    # ======================== TRUY VẤN ========================
    query = "Lợi ích của việc sử dụng Local LLM là gì?"
    print(f"\n❓ Query: {query}")
    
    result = await rag.aquery(
        query,
        mode="hybrid"
    )
    print(f"\n📝 Kết quả:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
