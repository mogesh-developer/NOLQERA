from nolqera.tokenization import Tokenizer


class Sentence:
    """Represents a single sentence from a document."""

    def __init__(self, text: str, index: int):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        self.text = text.strip()
        self.index = index

    def tokens(self) -> list[str]:
        """Tokenize the sentence."""
        tokenizer = Tokenizer()
        return tokenizer.tokenize(self.text)

    def __repr__(self) -> str:
        return f"Sentence(index={self.index}, text={self.text!r})"