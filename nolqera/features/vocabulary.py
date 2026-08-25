import json
from collections import Counter


class Vocabulary:
    """Manage token-to-index mappings for NLP features."""

    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        min_frequency: int = 1,
        add_unk: bool = False,
    ):
        if not isinstance(min_frequency, int):
            raise TypeError("min_frequency must be an integer")

        if min_frequency <= 0:
            raise ValueError(
                "min_frequency must be greater than 0"
            )

        if not isinstance(add_unk, bool):
            raise TypeError("add_unk must be a boolean")

        self.min_frequency = min_frequency
        self.add_unk = add_unk

        self.token_to_index: dict[str, int] = {}
        self.index_to_token: dict[int, str] = {}

    def fit(self, documents: list[list[str]]) -> None:
        """Build vocabulary from tokenized documents."""

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        frequencies = Counter(
            token
            for document in documents
            for token in document
        )

        tokens = sorted(
            token
            for token, frequency in frequencies.items()
            if frequency >= self.min_frequency
        )

        if self.add_unk:
            tokens = [
                self.UNK_TOKEN,
                *tokens,
            ]

        self.token_to_index = {
            token: index
            for index, token in enumerate(tokens)
        }

        self.index_to_token = {
            index: token
            for token, index in self.token_to_index.items()
        }

    @property
    def size(self) -> int:
        """Return vocabulary size."""

        return len(self.token_to_index)

    def contains(self, token: str) -> bool:
        """Check whether a token exists."""

        return token in self.token_to_index

    def get_index(self, token: str) -> int:
        """Return the index of a token."""

        if token in self.token_to_index:
            return self.token_to_index[token]

        if self.add_unk:
            return self.token_to_index[self.UNK_TOKEN]

        raise KeyError(f"Unknown token: {token}")

    def get_token(self, index: int) -> str:
        """Return the token at an index."""

        if index not in self.index_to_token:
            raise KeyError(f"Unknown index: {index}")

        return self.index_to_token[index]

    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens into integer IDs."""

        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")

        return [
            self.get_index(token)
            for token in tokens
        ]

    def decode(self, indices: list[int]) -> list[str]:
        """Convert integer IDs back into tokens."""

        if not isinstance(indices, list):
            raise TypeError("indices must be a list")

        return [
            self.get_token(index)
            for index in indices
        ]

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""

        if not isinstance(path, str):
            raise TypeError("path must be a string")

        data = {
            "min_frequency": self.min_frequency,
            "add_unk": self.add_unk,
            "token_to_index": self.token_to_index,
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load a vocabulary from a JSON file."""

        if not isinstance(path, str):
            raise TypeError("path must be a string")

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        vocabulary = cls(
            min_frequency=data["min_frequency"],
            add_unk=data["add_unk"],
        )

        vocabulary.token_to_index = {
            token: int(index)
            for token, index in data["token_to_index"].items()
        }

        vocabulary.index_to_token = {
            index: token
            for token, index in vocabulary.token_to_index.items()
        }

        return vocabulary