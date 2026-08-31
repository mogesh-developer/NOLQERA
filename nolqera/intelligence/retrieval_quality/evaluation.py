from __future__ import annotations

from typing import Iterable, List, Sequence


def precision_at_k(
    retrieved: Sequence[int],
    relevant: Iterable[int],
    k: int,
) -> float:
    _validate_k(k)

    retrieved = list(retrieved)
    relevant = set(relevant)

    if not retrieved or not relevant:
        return 0.0

    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        item in relevant
        for item in top_k
    )

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved: Sequence[int],
    relevant: Iterable[int],
    k: int,
) -> float:
    _validate_k(k)

    retrieved = list(retrieved)
    relevant = set(relevant)

    if not relevant:
        return 0.0

    top_k = retrieved[:k]

    relevant_count = sum(
        item in relevant
        for item in top_k
    )

    return relevant_count / len(relevant)


def hit_rate_at_k(
    retrieved: Sequence[int],
    relevant: Iterable[int],
    k: int,
) -> float:
    _validate_k(k)

    retrieved = list(retrieved)
    relevant = set(relevant)

    if not relevant:
        return 0.0

    return float(
        any(
            item in relevant
            for item in retrieved[:k]
        )
    )


def reciprocal_rank(
    retrieved: Sequence[int],
    relevant: Iterable[int],
) -> float:
    retrieved = list(retrieved)
    relevant = set(relevant)

    if not relevant:
        return 0.0

    for position, item in enumerate(
        retrieved,
        start=1,
    ):
        if item in relevant:
            return 1.0 / position

    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[int]],
    relevant_sets: Sequence[Iterable[int]],
) -> float:
    if len(rankings) != len(relevant_sets):
        raise ValueError(
            "rankings and relevant_sets "
            "must have the same length"
        )

    if not rankings:
        return 0.0

    scores = [
        reciprocal_rank(
            ranking,
            relevant,
        )
        for ranking, relevant in zip(
            rankings,
            relevant_sets,
        )
    ]

    return sum(scores) / len(scores)


def _validate_k(k: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than zero")