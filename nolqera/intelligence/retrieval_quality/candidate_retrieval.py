from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)
from nolqera.intelligence.semantic_search.service import (
    SemanticSearchService,
)


@dataclass(frozen=True)
class CandidateRetrievalResult:
    """
    Represents the candidate pool produced by retrieval.
    """

    results: List[SemanticSearchResult]

    @property
    def count(self) -> int:
        return len(self.results)


class CandidateRetriever:
    """
    First-stage retrieval layer.

    Retrieves a larger candidate pool from the existing
    SemanticSearchService so that later retrieval-quality
    stages can refine the candidates.
    """

    def __init__(self, service: SemanticSearchService):
        if not isinstance(service, SemanticSearchService):
            raise TypeError(
                "service must be a SemanticSearchService"
            )

        self._service = service

    @property
    def service(self) -> SemanticSearchService:
        return self._service

    def retrieve(
        self,
        query: str,
        candidate_k: int = 10,
    ) -> CandidateRetrievalResult:
        """
        Retrieve a candidate pool for the given query.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if not isinstance(candidate_k, int):
            raise TypeError(
                "candidate_k must be an integer"
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero"
            )

        results = self._service.search(
            query,
            top_k=candidate_k,
        )

        return CandidateRetrievalResult(
            results=list(results)
        )