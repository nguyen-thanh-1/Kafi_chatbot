def contains_cjk(text):
    """
    Detects if the text contains Chinese, Japanese, or Korean characters.
    Used to filter out hallucinated content from the model.
    """
    for char in text:
        # Check unicode ranges for CJK
        if '\u4e00' <= char <= '\u9fff':
            return True
            
    return False

def clean_response(text):
    """
    Optional: clean up response if needed.
    """
    return text.strip()
