def is_vietnamese(text):
    """
    Check if the text contains Vietnamese-specific characters.
    """
    vietnamese_chars = "ăâđêôơưĂÂĐÊÔƠƯàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ"
    return any(ch in text for ch in vietnamese_chars)

EN_KEYWORDS = [
    "grammar", "tense", "rewrite", "correct",
    "dịch", "translate", "ngữ pháp", "viết lại",
    "english", "vocabulary"
]

def route_language(text):
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
        
    # 3. Default to Vietnamese for this specific educational context
    return "vi"
