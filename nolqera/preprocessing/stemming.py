from collections.abc import Iterable


class SimpleStemmer:
    """A basic rule-based English stemmer."""

    def stem(self, word: str) -> str:
        """Reduce a single word to a basic stem."""

        if not isinstance(word, str):
            raise TypeError("word must be a string")

        word = word.lower()

        if len(word) <= 3:
            return word

        if word.endswith("ing") and len(word) > 5:
            return word[:-3]

        if word.endswith("ed") and len(word) > 4:
            return word[:-2]

        if word.endswith("ly") and len(word) > 4:
            return word[:-2]

        if word.endswith("es") and len(word) > 4:
            return word[:-2]

        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]

        return word

    def stem_tokens(self, tokens: Iterable[str]) -> list[str]:
        """Stem a sequence of tokens."""

        if not isinstance(tokens, (list, tuple)):
            raise TypeError("tokens must be a list or tuple")

        return [
            self.stem(token)
            for token in tokens
        ]