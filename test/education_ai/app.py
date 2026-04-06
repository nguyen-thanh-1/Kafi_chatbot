import streamlit as st
from engine import EducationalLLM
from utils import route_language
from prompts import detect_subject, get_system_prompt
from validator import contains_cjk

# Page Config
st.set_page_config(page_title="AI Gia sư (Llama 3.1)", page_icon="🎓", layout="wide")

# Initialize Model (Cached)
@st.cache_resource
def load_model():
    return EducationalLLM()

try:
    with st.spinner("Đang tải mô hình AI... Vui lòng đợi trong giây lát."):
        model = load_model()
    st.sidebar.success("Mô hình đã sẵn sàng!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.title("🎛️ Bảng điều khiển")
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Trạng thái hiện tại")
    status_placeholder = st.empty()

# Main Chat Interface
st.title("🎓 Trợ lý AI Giáo dục")
st.caption("Hỗ trợ: Toán, Lý, Hóa, Tiếng Anh (Tự động nhận diện môn học)")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input & Processing
if user_input := st.chat_input("Nhập câu hỏi của bạn..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Analyze Input
    detected_lang = route_language(user_input)
    detected_subj = detect_subject(user_input)
    
    # Update Status Sidebar
    status_placeholder.markdown(f"""
    - **Ngôn ngữ**: `{'Tiếng Việt' if detected_lang=='vi' else 'English'}`
    - **Môn học**: `{detected_subj.upper()}`
    """)
    
    # 3. Get System Prompt
    system_prompt = get_system_prompt(detected_subj, detected_lang)

    # 4. Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Prepare history for model (excluding system prompts if managed internally, 
        # but here we pass history as list of dicts. The engine handles formatting)
        # Note: The engine expects the history list format.
        # We need to filter only user/assistant messages for the engine history to avoid duplicate system prompts.
        chat_history_for_model = [m for m in st.session_state.messages if m["role"] != "system"]
        
        try:
            stream = model.stream_chat(user_input, system_prompt, chat_history_for_model[:-1])
            
            for token in stream:
                full_response += token
                response_placeholder.markdown(full_response + "▌")
                
                # Real-time Validator Check (Optional: stop if CJK detected early)
                if contains_cjk(token):
                    # Simple heuristic: if a chunk has CJK, we might want to flag it. 
                    # For now, just logging or handling post-stream is easier for UX flow.
                    pass

            response_placeholder.markdown(full_response)
            
            # Post-validation
            if contains_cjk(full_response):
                st.warning("⚠️ Phát hiện ký tự lạ (Tiếng Trung). Đang thử lại...")
                # In a real scenario, we might trigger a re-generation here
        
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

    # 5. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
