from .sentence_tokenizer import split_sentences
from .word_tokenizer import tokenize_words


class Tokenizer:
    """Unified tokenizer interface."""

    def sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        return split_sentences(text)

    def words(self, text: str) -> list[str]:
        """Split text into word and punctuation tokens."""
        return tokenize_words(text)