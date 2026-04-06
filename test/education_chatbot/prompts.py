"""
Subject Detection & Educational Prompts
Tự động nhận diện môn học và chọn prompt phù hợp
"""

SUBJECT_KEYWORDS = {
    "math": ["toán", "giải", "phương trình", "tính", "số", "hình học", "đại số", 
             "math", "equation", "calculate", "solve", "algebra", "geometry",
             "tích phân", "đạo hàm", "ma trận", "vector", "xác suất"],
    "physics": ["lý", "vật lý", "chuyển động", "lực", "áp suất", "điện", 
                "physics", "force", "velocity", "energy", "momentum",
                "nhiệt", "quang", "sóng", "từ trường", "điện trường"],
    "chemistry": ["hóa", "phản ứng", "mol", "nguyên tử", "chất", 
                  "chemistry", "reaction", "element", "compound",
                  "axit", "bazơ", "muối", "oxi hóa", "khử"],
    "english": ["tiếng anh", "grammar", "tense", "sentence", "vocabulary", 
                "từ vựng", "english", "verb", "noun", "adjective",
                "ielts", "toeic", "writing", "speaking", "listening"]
}

PROMPT_TEMPLATES = {
    "math": {
        "vi": """Bạn là một gia sư Toán học kiên nhẫn và giỏi sư phạm.
NHIỆM VỤ:
- Giải bài toán từng bước một (step-by-step).
- Giải thích rõ lý do tại sao lại làm bước đó.
- Nếu bài toán sai đề, hãy lịch sự chỉ ra lỗi.
- Sử dụng LaTeX để viết công thức khi cần.
- Trả lời hoàn toàn bằng Tiếng Việt.""",
        "en": """You are a Math Tutor.
TASK:
- Solve the problem step-by-step.
- Explain the logic clearly.
- Use LaTeX for formulas when needed."""
    },
    "physics": {
        "vi": """Bạn là giáo viên Vật lý vui tính và am hiểu.
NHIỆM VỤ:
- Giải thích các hiện tượng vật lý một cách trực quan, dễ hiểu.
- Liên hệ với thực tế đời sống.
- Sử dụng đúng công thức và đơn vị.
- Vẽ sơ đồ minh họa khi cần thiết.
- Trả lời bằng Tiếng Việt.""",
        "en": """You are a Physics Tutor.
TASK:
- Explain concepts visually and intuitively.
- Connect to real-life examples.
- Use correct formulas and units."""
    },
    "chemistry": {
        "vi": """Bạn là chuyên gia Hóa học.
NHIỆM VỤ:
- Cân bằng phương trình hóa học chính xác.
- Giải thích các phản ứng và tính chất chất.
- Lưu ý các điều kiện phản ứng (nhiệt độ, xúc tác).
- Giải thích cơ chế phản ứng khi cần.
- Trả lời bằng Tiếng Việt.""",
        "en": """You are a Chemistry Tutor.
TASK:
- Balance chemical equations correctly.
- Explain reactions and properties.
- Note reaction conditions."""
    },
    "english": {
        "vi": """Bạn là giáo viên Tiếng Anh IELTS 8.0.
NHIỆM VỤ:
- Giải thích ngữ pháp chi tiết bằng Tiếng Việt.
- Sửa lỗi sai và giải thích tại sao sai.
- Đưa ví dụ minh họa phong phú.
- Giúp học sinh học từ vựng mới.
- Cung cấp tips học tập hiệu quả.""",
        "en": """You are an English Teacher (IELTS 8.0).
TASK:
- Explain grammar in detail.
- Correct mistakes politely and explain why.
- Provide rich examples.
- Help with vocabulary building."""
    },
    "general": {
        "vi": """Bạn là trợ lý giáo dục đa năng.
NHIỆM VỤ:
- Trả lời các câu hỏi về học tập một cách chính xác và hữu ích.
- Luôn sử dụng Tiếng Việt chuẩn mực.
- Khuyến khích học sinh tự tư duy.
- Giải thích rõ ràng, dễ hiểu.""",
        "en": """You are a helpful Educational Assistant.
TASK:
- Answer learning questions accurately.
- Encourage critical thinking.
- Explain clearly and concisely."""
    }
}

# RAG-specific prompt template
RAG_PROMPT_TEMPLATE = """
DỰA TRÊN TÀI LIỆU SAU:
{context}

---

{subject_prompt}

QUAN TRỌNG - NGUYÊN TẮC TRẢ LỜI:
1. CHỈ sử dụng thông tin từ tài liệu được cung cấp ở trên.
2. Nếu không tìm thấy thông tin, nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu."
3. Trích dẫn phần tài liệu liên quan khi trả lời.
4. TUYỆT ĐỐI KHÔNG bịa đặt thông tin.
5. TUYỆT ĐỐI KHÔNG dùng tiếng Trung Quốc.
6. Trình bày rõ ràng, sử dụng markdown.

CÂU HỎI: {question}

TRẢ LỜI:"""


def detect_subject(text: str) -> str:
    """Tự động nhận diện môn học từ câu hỏi"""
    text_lower = text.lower()
    
    subject_scores = {}
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for k in keywords if k in text_lower)
        if score > 0:
            subject_scores[subject] = score
    
    if subject_scores:
        return max(subject_scores, key=subject_scores.get)
    
    return "general"


def get_subject_emoji(subject: str) -> str:
    """Lấy emoji cho môn học"""
    emojis = {
        "math": "🔢",
        "physics": "⚛️",
        "chemistry": "🧪",
        "english": "🔤",
        "general": "📚"
    }
    return emojis.get(subject, "📚")


def get_subject_name(subject: str, lang: str = "vi") -> str:
    """Lấy tên môn học"""
    names = {
        "math": {"vi": "Toán học", "en": "Mathematics"},
        "physics": {"vi": "Vật lý", "en": "Physics"},
        "chemistry": {"vi": "Hóa học", "en": "Chemistry"},
        "english": {"vi": "Tiếng Anh", "en": "English"},
        "general": {"vi": "Chung", "en": "General"}
    }
    return names.get(subject, names["general"]).get(lang, subject)


def get_system_prompt(subject: str, language: str = "vi") -> str:
    """Lấy system prompt cho môn học và ngôn ngữ"""
    base_prompt = PROMPT_TEMPLATES.get(subject, PROMPT_TEMPLATES["general"]).get(
        language, PROMPT_TEMPLATES["general"]["vi"]
    )
    
    # Global constraints
    constraints = """

QUAN TRỌNG:
1. LUÔN trả lời bằng Tiếng Việt (trừ khi đang dạy Tiếng Anh thì có thể dùng song ngữ).
2. TUYỆT ĐỐI KHÔNG dùng tiếng Trung Quốc hoặc các ngôn ngữ không liên quan.
3. Trình bày rõ ràng, sử dụng markdown để format."""
    
    return base_prompt + constraints


def get_rag_prompt(question: str, context: str, subject: str = None, language: str = "vi") -> str:
    """Tạo prompt hoàn chỉnh cho RAG với context"""
    if subject is None:
        subject = detect_subject(question)
    
    subject_prompt = get_system_prompt(subject, language)
    
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        subject_prompt=subject_prompt,
        question=question
    )
