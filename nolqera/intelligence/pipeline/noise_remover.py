from nolqera.intelligence.context_optimization.noise_detection import (
    NoiseDetector,
)
from nolqera.intelligence.semantic_search.models import SemanticSearchResult


class NoiseRemover:
    """
    Phase 4 pipeline adapter for noise removal.

    Delegates noise detection and filtering to the existing
    NoiseDetector from the Context Optimization layer.
    """

    def __init__(self, noise_detector: NoiseDetector):
        if not isinstance(noise_detector, NoiseDetector):
            raise TypeError(
                "noise_detector must be a NoiseDetector"
            )

        self._noise_detector = noise_detector

    def remove(
        self,
        results: list[SemanticSearchResult],
    ) -> list[SemanticSearchResult]:
        if not isinstance(results, list):
            raise TypeError("results must be a list")

        for result in results:
            if not isinstance(result, SemanticSearchResult):
                raise TypeError(
                    "all results must be SemanticSearchResult instances"
                )

        return self._noise_detector.filter(results)