"""
RAG Configuration - Cấu hình chung cho hệ thống RAG
Dùng chung cho index_docs.py và query_rag.py
"""

import os

# ======================== PATHS ========================
# Folder chứa tài liệu nguồn
COURSES_FOLDER = "./courses"

# Folder output cho parsed documents (cache)
OUTPUT_DIR = "./output_courses"

# Folder lưu RAG database (vector DB + knowledge graph)
RAG_STORAGE = "./rag_storage_courses"

# File lưu danh sách đã index (cho incremental indexing)
INDEX_TRACKER_FILE = os.path.join(RAG_STORAGE, "indexed_files.json")


# ======================== SUPPORTED FORMATS ========================
SUPPORTED_EXTENSIONS = [
    ".pdf",   # PDF documents
    ".txt",   # Plain text
    ".docx",  # Microsoft Word
    ".doc",   # Microsoft Word (legacy)
    ".xlsx",  # Microsoft Excel
    ".xls",   # Microsoft Excel (legacy)
    ".csv",   # CSV files
    ".pptx",  # PowerPoint
    ".ppt",   # PowerPoint (legacy)
    ".md",    # Markdown
]


# ======================== MODEL CONFIGURATION ========================
# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 256 #256

# LLM settings
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.1


# ======================== RAG CONFIGURATION ========================
RAG_CONFIG = {
    "working_dir": RAG_STORAGE,
    "parser": "txt",           # Parser: mineru hoặc docling 
    "parse_method": "auto",       # auto, ocr, hoặc txt
    "enable_image_processing": False,  # Tắt để tăng tốc
    "enable_table_processing": True,   # Bật cho Excel/CSV / True
    "enable_equation_processing": False,
}


# ======================== HELPER FUNCTIONS ========================
def ensure_directories():
    """Tạo các thư mục cần thiết nếu chưa tồn tại"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAG_STORAGE, exist_ok=True)


def get_supported_files(folder: str) -> list:
    """Lấy danh sách files được hỗ trợ trong folder"""
    if not os.path.exists(folder):
        return []
    
    files = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def get_file_info(folder: str, filename: str) -> dict:
    """Lấy thông tin file (size, modified time)"""
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        return None
    
    stat = os.stat(file_path)
    return {
        "filename": filename,
        "path": file_path,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified_time": stat.st_mtime,
    }
