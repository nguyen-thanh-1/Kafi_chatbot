"""
Utilities - Language Routing & Text Processing
"""

# Vietnamese-specific characters
VIETNAMESE_CHARS = "ăâđêôơưĂÂĐÊÔƠƯàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹầấậẩẫềếệểễồốộổỗừứựửữờớợởỡ"

# English learning keywords
EN_KEYWORDS = [
    "grammar", "tense", "rewrite", "correct",
    "dịch", "translate", "ngữ pháp", "viết lại",
    "english", "vocabulary", "ielts", "toeic"
]


def is_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese-specific characters"""
    return any(ch in text for ch in VIETNAMESE_CHARS)


def route_language(text: str) -> str:
    """
    Determine user language based on text analysis.
    Returns: 'vi' or 'en'
    """
    # 1. Prioritize explicit Vietnamese chars
    if is_vietnamese(text):
        return "vi"
    
    # 2. Check for English learning keywords
    input_lower = text.lower()
    if any(k in input_lower for k in EN_KEYWORDS):
        return "en"
    
    # 3. Default to Vietnamese for this educational context
    return "vi"


def contains_cjk(text: str) -> bool:
    """
    Check if text contains CJK (Chinese/Japanese/Korean) characters.
    Used to detect unwanted language mixing from LLM.
    """
    for char in text:
        code_point = ord(char)
        # CJK Unified Ideographs
        if 0x4E00 <= code_point <= 0x9FFF:
            return True
        # CJK Extension A
        if 0x3400 <= code_point <= 0x4DBF:
            return True
        # Japanese Hiragana
        if 0x3040 <= code_point <= 0x309F:
            return True
        # Japanese Katakana
        if 0x30A0 <= code_point <= 0x30FF:
            return True
        # Korean Hangul
        if 0xAC00 <= code_point <= 0xD7AF:
            return True
    return False


def clean_response(text: str) -> str:
    """Clean up LLM response - remove unwanted artifacts"""
    # Remove excessive newlines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    
    return text.strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
