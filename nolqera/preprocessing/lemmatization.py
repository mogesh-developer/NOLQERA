class SimpleLemmatizer:
    """A basic rule-based English lemmatizer."""

    def lemmatize(self, word: str) -> str:
        """Convert a word to a basic lemma."""

        if not isinstance(word, str):
            raise TypeError("word must be a string")

        word = word.lower()

        irregular_forms = {
            "was": "be",
            "were": "be",
            "is": "be",
            "are": "be",
            "am": "be",
            "went": "go",
            "gone": "go",
            "better": "good",
            "children": "child",
            "mice": "mouse",
        }

        if word in irregular_forms:
            return irregular_forms[word]

        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"

        if word.endswith("ing") and len(word) > 5:
            return word[:-3]

        if word.endswith("ed") and len(word) > 4:
            return word[:-2]

        if word.endswith("es") and len(word) > 4:
            return word[:-2]

        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]

        return word

    def lemmatize_tokens(
        self,
        tokens: list[str],
    ) -> list[str]:
        """Lemmatize a sequence of tokens."""

        if not isinstance(tokens, list):
            raise TypeError("tokens must be a list")

        return [
            self.lemmatize(token)
            for token in tokens
        ]