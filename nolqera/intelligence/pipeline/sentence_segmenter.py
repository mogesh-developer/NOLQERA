class SentenceSegmenter:
    """
    Splits normalized text into individual sentences.
    """

    def segment(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        sentences = []
        current = []

        for character in text:
            current.append(character)

            if character in ".!?":
                sentence = "".join(current).strip()

                if sentence:
                    sentences.append(sentence)

                current = []

        # Preserve trailing text without punctuation.
        trailing = "".join(current).strip()

        if trailing:
            sentences.append(trailing)

        return sentences