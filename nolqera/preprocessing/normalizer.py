def lowercase(text: str) -> str:
    """Convert text to lowercase"""
    return text.lower()

def normalize_whitespace(text : str) -> str:
    """Normalize whitespace in text"""
    return " ".join(text.split())

def normalize_text(text: str) -> str:
    """Normalize text"""
    text = lowercase(text)
    text = normalize_whitespace(text)
    return text

def normalize(text : str) -> str:
    """Apply basic text normalization"""
    text = lowercase(text)
    text = normalize_whitespace(text)
    return text