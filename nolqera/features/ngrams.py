from ..utils.text_utils import validate_n, validate_tokens


def generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """
    Generate n-grams from a sequence of tokens.
    """

    validate_tokens(tokens)
    validate_n(n)

    if n > len(tokens):
        return []

    return [
        tuple(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    ]