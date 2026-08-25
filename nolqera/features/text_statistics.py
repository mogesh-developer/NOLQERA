import re
from ..tokenization.word_tokenizer import tokenize_words

class TextStatistics:
    """Calculate statistical properties of text."""

    def __init__(self):
        self.text = ""
        self.tokens: list[str] = []
        self.sentences: list[str] = []

    def fit(
        self,
        text: str,
        tokens: list[str] | None = None,
    ) -> None:
        """Analyze a text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        self.text = text

        if tokens is not None:
            if not isinstance(tokens, list):
                raise TypeError("tokens must be a list")

            for token in tokens:
                if not isinstance(token, str):
                    raise TypeError(
                        "tokens must be strings"
                )

            self.tokens = tokens
        else:
            self.tokens = tokenize_words(text)

        self.sentences = self._split_sentences(text)

    @classmethod
    def analyze(
        cls,
        text: str,
        tokens: list[str] | None = None,
    ) -> "TextStatistics":
        """Create statistics for a text."""

        instance = cls()
        instance.fit(text, tokens=tokens)

        return instance
        
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into simple sentences."""

        if not text.strip():
            return []

        sentences = re.split(
            r"[.!?]+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @property
    def character_count(self) -> int:
        """Return total number of characters."""

        return len(self.text)

    @property
    def character_count_no_spaces(self) -> int:
        """Return characters excluding whitespace."""

        return len(
            "".join(self.text.split())
        )

    @property
    def token_count(self) -> int:
        """Return total token count."""

        return len(self.tokens)

    @property
    def unique_token_count(self) -> int:
        """Return number of unique tokens."""

        return len(set(self.tokens))

    @property
    def sentence_count(self) -> int:
        """Return number of sentences."""

        return len(self.sentences)

    @property
    def average_token_length(self) -> float:
        """Return average token length."""

        if not self.tokens:
            return 0.0

        return sum(
            len(token)
            for token in self.tokens
        ) / len(self.tokens)

    @property
    def vocabulary_richness(self) -> float:
        """Return type-token ratio."""

        if not self.tokens:
            return 0.0

        return (
            self.unique_token_count
            / self.token_count
        )

    def summary(self) -> dict[str, int | float]:
        """Return a summary of text statistics."""

        return {
            "characters": self.character_count,
            "characters_no_spaces": self.character_count_no_spaces,
            "tokens": self.token_count,
            "unique_tokens": self.unique_token_count,
            "sentences": self.sentence_count,
            "average_token_length": self.average_token_length,
            "vocabulary_richness": self.vocabulary_richness,
        }

    def to_dict(self) -> dict[str, int | float]:
        """Return statistics as a dictionary."""

        return self.summary()