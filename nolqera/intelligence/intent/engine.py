from .candidates import IntentCandidateExtractor
from .classifier import IntentClassifier
from .models import IntentResult
from .ranking import IntentRanker
from .scorer import IntentScorer


class IntentEngine:
    """
    End-to-end intent intelligence engine.

    Pipeline:
        text
        -> candidate extraction
        -> classification
        -> scoring
        -> ranking
        -> public IntentResult objects
    """

    def __init__(
        self,
        extractor: IntentCandidateExtractor | None = None,
        classifier: IntentClassifier | None = None,
        scorer: IntentScorer | None = None,
        ranker: IntentRanker | None = None,
    ) -> None:

        self.extractor = (
            extractor
            or IntentCandidateExtractor()
        )

        self.classifier = (
            classifier
            or IntentClassifier()
        )

        self.scorer = (
            scorer
            or IntentScorer()
        )

        self.ranker = (
            ranker
            or IntentRanker()
        )

    def analyze(
        self,
        text: str,
    ) -> list[IntentResult]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        # 1. Extract intent signals
        candidates = self.extractor.extract(text)

        if not candidates:
            return []

        # 2. Classify signals
        classifications = self.classifier.classify(
            candidates
        )

        if not classifications:
            return []

        # 3. Calculate intent confidence
        scores = self.scorer.score(
            classifications
        )

        if not scores:
            return []

        # 4. Rank intents
        ranked = self.ranker.rank(
            scores
        )

        # 5. Convert to public models
        return [
            IntentResult(
                intent=item.intent,
                score=item.score,
                evidence_count=item.evidence_count,
            )
            for item in ranked
        ]