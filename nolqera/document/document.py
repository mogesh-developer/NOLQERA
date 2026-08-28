from .sentence import Sentence


class Document:
    """Represents a document inside NOLQERA."""

    def __init__(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        self.text = text
        self.metadata = {}

    def sentences(self) -> list[Sentence]:
        """Split the document into sentences."""
        raw_sentences = self.text.replace("!", ".").replace("?", ".").split(".")

        sentences = []
        for index, text in enumerate(raw_sentences):
            text = text.strip()
            if text:
                sentences.append(Sentence(text, index))

        return sentences

    def __repr__(self) -> str:
        return f"Document(length={len(self.text)})"