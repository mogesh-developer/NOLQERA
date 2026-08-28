from math import sqrt


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    if not isinstance(vector_a, list):
        raise TypeError("vector_a must be a list")

    if not isinstance(vector_b, list):
        raise TypeError("vector_b must be a list")

    if not vector_a or not vector_b:
        raise ValueError("vectors cannot be empty")

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "vectors must have the same dimensions"
        )

    if any(
        not isinstance(value, (int, float))
        for value in vector_a + vector_b
    ):
        raise TypeError(
            "vectors must contain numeric values"
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )