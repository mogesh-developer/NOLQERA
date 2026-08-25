import re

from ..utils.text_utils import validate_text


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using common sentence-ending punctuation.
    """

    validate_text(text)

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]