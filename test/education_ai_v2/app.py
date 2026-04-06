"""
🎓 Education AI v2 - Trợ lý Giáo dục thông minh với RAG
═══════════════════════════════════════════════════════════
Features:
- Upload tài liệu (PDF, TXT, MD, CSV)
- Tự động xử lý và index
- Hỏi đáp dựa trên tài liệu (RAG)
- Streaming response
- Hiển thị trạng thái xử lý real-time

Chạy: streamlit run app.py
═══════════════════════════════════════════════════════════
"""

import streamlit as st
import os
import time
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Education AI v2",
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
    .file-card b {
        color: #00d4ff;
    }
    .file-card small {
        color: #a0a0a0;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    /* Dark sidebar */
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
os.makedirs(UPLOADS_DIR, exist_ok=True)

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
    """Load RAG Engine (embedding + reranker)"""
    from rag_engine import get_rag_engine
    return get_rag_engine()


@st.cache_resource(show_spinner=False)
def load_llm():
    """Load LLM"""
    from engine_llm import EducationalLLM
    return EducationalLLM()


# ═══════════════════════════════════════════════════════════
# SIDEBAR - File Upload & Status
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎓 Education AI v2")
    st.caption("Trợ lý học tập thông minh với RAG")
    
    st.markdown("---")
    
    # Model Loading Status
    st.markdown("### 🔧 Trạng thái hệ thống")
    
    if not st.session_state.models_loaded:
        with st.spinner("Đang tải models..."):
            try:
                rag_engine = load_rag_engine()
                llm = load_llm()
                st.session_state.models_loaded = True
                st.success("✅ Hệ thống sẵn sàng!")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
                st.stop()
    else:
        rag_engine = load_rag_engine()
        llm = load_llm()
        st.success("✅ Hệ thống sẵn sàng!")
    
    # Stats
    stats = rag_engine.get_stats()
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
                # Update status
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
                    file_info = rag_engine.add_file(filepath, file.name)
                    rag_engine.process_queue()
                    elapsed = time.time() - start_time
                    
                    # Update status
                    st.session_state.file_status[file.name] = {
                        "status": "completed",
                        "chunks": rag_engine.vector_store.files.get(file.name, {}).get('chunks', 0),
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
st.title("💬 Hỏi đáp với Tài liệu")

# Instructions
if not st.session_state.file_status:
    st.info("""
    🎓 **Education AI v2 - Trợ lý Học tập Thông minh**
    
    **Bạn có thể hỏi ngay!** Không cần upload tài liệu.
    
    **Tính năng:**
    - 💬 **Chat Mode**: Hỏi đáp trực tiếp với AI
    - 📚 **RAG Mode**: Upload tài liệu → AI trả lời dựa trên nội dung
    - 🔢 Toán học | ⚛️ Vật lý | 🧪 Hóa học | 🔤 Tiếng Anh
    - 🤖 Tự động nhận diện môn học và ngôn ngữ
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
    
    # Analyze question (subject & language detection)
    analysis = llm.analyze_question(user_input)
    
    # Determine mode: RAG or Normal Chat
    has_documents = bool(st.session_state.file_status)
    use_rag = False
    context_chunks = []
    
    if has_documents:
        with st.spinner("🔍 Đang tìm kiếm trong tài liệu..."):
            result = rag_engine.query(user_input)
            if isinstance(result, list) and len(result) > 0:
                context_chunks = result
                use_rag = True
    
    # Show detection info with mode indicator
    mode_text = "📚 RAG Mode" if use_rag else "💬 Chat Mode"
    mode_color = "#00d4ff" if use_rag else "#FFD700"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 10px 15px; border-radius: 8px; margin: 10px 0;
                border-left: 4px solid {mode_color};">
        {analysis['subject_emoji']} <b>Môn học:</b> {analysis['subject_name']} | 
        🌐 <b>Ngôn ngữ:</b> {analysis['language_name']} |
        <span style="color: {mode_color};">{mode_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            if use_rag:
                # RAG Mode: Answer with context
                for token in llm.answer_with_context(user_input, context_chunks, stream=True):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
            else:
                # Normal Chat Mode: Use LLM directly (like education_ai)
                history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages[:-1]  # Exclude current message
                ][-10:]  # Last 10 messages for context
                
                for token in llm.chat_without_rag(user_input, history, stream=True):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"❌ Lỗi khi xử lý: {e}"
            response_placeholder.markdown(full_response)
        
        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.caption("""
🎓 **Education AI v2** | Powered by Llama 3.1 8B + BGE-M3 + Reranker  
📚 Upload tài liệu → Tự động xử lý → Hỏi đáp thông minh
""")
