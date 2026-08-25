import re

from ..utils.text_utils import validate_text


TOKEN_PATTERN = re.compile(
    r"""
    \d+(?:\.\d+)?
    |
    [^\W\d_]+(?:'[^\W\d_]+)*
    |
    [^\w\s]
    """,
    re.VERBOSE | re.UNICODE,
)


def tokenize_words(text: str) -> list[str]:
    """
    Tokenize text into words, numbers, contractions,
    and punctuation tokens.
    """

    validate_text(text)

    if not text.strip():
        return []

    return TOKEN_PATTERN.findall(text)