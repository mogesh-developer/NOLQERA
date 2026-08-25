def validate_text(text: str) -> None:
    """Validate a text input."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

def validate_text(text: str) -> None:
    """Validate a text input."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")


def validate_tokens(tokens: list[str]) -> None:
    """Validate a token sequence."""

    if not isinstance(tokens, list):
        raise TypeError("tokens must be a list")

    if not all(isinstance(token, str) for token in tokens):
        raise TypeError("every token must be a string")


def validate_n(n: int) -> None:
    """Validate an n-gram size."""

    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n <= 0:
        raise ValueError("n must be greater than 0")