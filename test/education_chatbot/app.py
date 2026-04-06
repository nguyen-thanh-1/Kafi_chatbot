"""
🎓 Education Chatbot - Complete RAG System
═══════════════════════════════════════════════════════════

FEATURES:
- Streamlit UI with upload & chat
- 2-Stage RAG (Question Routing)
- Subject Detection (Math, Physics, Chemistry, English)
- Hybrid prompts (context + LLM knowledge)
- Strict mode for document-specific questions

COMPONENTS:
- Embedding: all-MiniLM-L6-v2 (fast)
- Reranker: FlashRank (ONNX)
- LLM: Llama 3.1 8B (4-bit)

RUN: streamlit run app.py

═══════════════════════════════════════════════════════════
"""

import streamlit as st
import os
import time
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Education Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme
st.markdown("""
<style>
    .status-pending { color: #FFA500; font-weight: bold; }
    .status-processing { color: #00BFFF; font-weight: bold; }
    .status-completed { color: #00FF7F; font-weight: bold; }
    .status-error { color: #FF6B6B; font-weight: bold; }
    .file-card {
        padding: 12px 15px;
        border-radius: 10px;
        margin: 8px 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        color: #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .file-card b { color: #00d4ff; }
    .file-card small { color: #a0a0a0; }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .mode-hybrid {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
    }
    .mode-strict {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    .mode-llm {
        background: linear-gradient(135deg, #ffd700, #ffb700);
        color: #333;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# INITIALIZE
# ═══════════════════════════════════════════════════════════
UPLOADS_DIR = "./uploads"
DATA_DIR = "./data"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_status" not in st.session_state:
    st.session_state.file_status = {}

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False


# ═══════════════════════════════════════════════════════════
# LOAD MODELS (Cached)
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    """Load RAG Hybrid Engine"""
    from rag_engine import RAGHybrid
    return RAGHybrid()


@st.cache_resource(show_spinner=False)
def load_prompts():
    """Load prompt functions"""
    from prompts import detect_subject, get_subject_emoji, get_subject_name
    return {
        "detect_subject": detect_subject,
        "get_subject_emoji": get_subject_emoji,
        "get_subject_name": get_subject_name
    }


@st.cache_resource(show_spinner=False)
def load_utils():
    """Load utility functions"""
    from utils import route_language, contains_cjk, clean_response
    return {
        "route_language": route_language,
        "contains_cjk": contains_cjk, 
        "clean_response": clean_response
    }


# ═══════════════════════════════════════════════════════════
# SIDEBAR - File Upload & Status
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎓 Education Chatbot")
    st.caption("2-Stage RAG với Question Routing")
    
    st.markdown("---")
    
    # Model Loading Status
    st.markdown("### 🔧 Trạng thái hệ thống")
    
    if not st.session_state.models_loaded:
        with st.spinner("Đang tải models..."):
            try:
                rag = load_rag_engine()
                prompts = load_prompts()
                utils = load_utils()
                
                # Preload models
                rag.preload_lite()
                
                st.session_state.models_loaded = True
                st.success("✅ Hệ thống sẵn sàng!")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
                st.stop()
    else:
        rag = load_rag_engine()
        prompts = load_prompts()
        utils = load_utils()
        st.success("✅ Hệ thống sẵn sàng!")
    
    # Stats
    stats = rag.get_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📁 Files", stats['files'])
    with col2:
        st.metric("📦 Chunks", stats['chunks'])
    
    st.markdown("---")
    
    # File Upload
    st.markdown("### 📤 Upload tài liệu")
    uploaded_files = st.file_uploader(
        "Kéo thả hoặc chọn files",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Process uploaded files
    if uploaded_files:
        new_files = []
        for file in uploaded_files:
            if file.name not in st.session_state.file_status:
                new_files.append(file)
        
        if new_files:
            st.markdown("#### 🔄 Đang xử lý...")
            progress_bar = st.progress(0)
            
            for i, file in enumerate(new_files):
                st.session_state.file_status[file.name] = {
                    "status": "processing",
                    "chunks": 0,
                    "time": None
                }
                
                # Save file
                filepath = os.path.join(UPLOADS_DIR, file.name)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
                
                # Process
                try:
                    start_time = time.time()
                    chunks = rag.index_file(filepath, file.name)
                    elapsed = time.time() - start_time
                    
                    st.session_state.file_status[file.name] = {
                        "status": "completed",
                        "chunks": chunks,
                        "time": f"{elapsed:.1f}s"
                    }
                except Exception as e:
                    st.session_state.file_status[file.name] = {
                        "status": "error",
                        "error": str(e)
                    }
                
                progress_bar.progress((i + 1) / len(new_files))
            
            progress_bar.empty()
            st.rerun()
    
    # File Status List
    if st.session_state.file_status:
        st.markdown("#### 📋 Danh sách tài liệu")
        
        for filename, info in st.session_state.file_status.items():
            status = info.get("status", "pending")
            
            if status == "completed":
                chunks = info.get('chunks', 0)
                time_taken = info.get('time', '')
                st.markdown(f"""
                <div class="file-card">
                    ✅ <b>{filename}</b><br/>
                    <small>📦 {chunks} chunks | ⏱️ {time_taken}</small>
                </div>
                """, unsafe_allow_html=True)
            elif status == "processing":
                st.markdown(f"""
                <div class="file-card">
                    🔄 <b>{filename}</b><br/>
                    <small class="status-processing">Đang xử lý...</small>
                </div>
                """, unsafe_allow_html=True)
            elif status == "error":
                error = info.get('error', 'Unknown error')
                st.markdown(f"""
                <div class="file-card">
                    ❌ <b>{filename}</b><br/>
                    <small class="status-error">{error[:50]}</small>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xóa chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()


# ═══════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════
st.title("💬 Hỏi đáp Thông minh")

# Instructions
if not st.session_state.file_status:
    st.info("""
    🎓 **Education Chatbot - 2-Stage RAG System**
    
    **Bạn có thể hỏi ngay!** Không cần upload tài liệu.
    
    **Hệ thống tự động chọn mode:**
    - ⚡ **Fast Mode**: Câu hỏi chung → RAG Lite + LLM General Knowledge
    - 📚 **Deep Mode**: "Theo tài liệu...", "Trong sách..." → RAG Pro (strict)
    
    **Môn học hỗ trợ:**
    🔢 Toán học | ⚛️ Vật lý | 🧪 Hóa học | 🔤 Tiếng Anh
    """)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Nhập câu hỏi..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Analyze question
    detect_subject = prompts["detect_subject"]
    get_subject_emoji = prompts["get_subject_emoji"]
    get_subject_name = prompts["get_subject_name"]
    route_language = utils["route_language"]
    
    subject = detect_subject(user_input)
    language = route_language(user_input)
    subject_emoji = get_subject_emoji(subject)
    subject_name = get_subject_name(subject, language)
    
    # Show detection info
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 10px 15px; border-radius: 8px; margin: 10px 0;
                border-left: 4px solid #00d4ff;">
        {subject_emoji} <b>Môn học:</b> {subject_name} | 
        🌐 <b>Ngôn ngữ:</b> {"Tiếng Việt" if language == "vi" else "English"}
    </div>
    """, unsafe_allow_html=True)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            with st.spinner("🔍 Đang xử lý..."):
                # Query with routing
                answer, mode = rag.query_with_mode(user_input)
                
                # Show mode indicator
                if mode == "rag_lite":
                    mode_html = '<span class="mode-badge mode-hybrid">⚡ Hybrid Mode</span>'
                elif mode == "rag_pro":
                    mode_html = '<span class="mode-badge mode-strict">📚 Strict Mode</span>'
                else:
                    mode_html = '<span class="mode-badge mode-llm">🤖 LLM Only</span>'
                
                st.markdown(mode_html, unsafe_allow_html=True)
                response_placeholder.markdown(answer)
                
        except Exception as e:
            answer = f"❌ Lỗi khi xử lý: {e}"
            response_placeholder.markdown(answer)
        
        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.caption("""
🎓 **Education Chatbot** | Powered by 2-Stage RAG + Llama 3.1 8B  
⚡ Fast: MiniLM + FlashRank | 📚 Deep: BGE-M3 + Reranker
""")
