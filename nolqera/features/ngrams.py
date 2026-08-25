from ..tokenization.word_tokenizer import tokenize_words


def generate_ngrams(
    tokens: list[str],
    n: int,
) -> list[tuple[str, ...]]:
    """Generate n-grams from a sequence of tokens."""

    if not isinstance(tokens, list):
        raise TypeError("tokens must be a list")

    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n <= 0:
        raise ValueError("n must be greater than 0")

    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("tokens must contain strings")

    if n > len(tokens):
        return []

    return [
        tuple(tokens[index:index + n])
        for index in range(
            len(tokens) - n + 1
        )
    ]


def generate_ngrams_from_text(
    text: str,
    n: int,
) -> list[tuple[str, ...]]:
    """Generate n-grams directly from raw text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens = tokenize_words(text)

    return generate_ngrams(tokens, n)