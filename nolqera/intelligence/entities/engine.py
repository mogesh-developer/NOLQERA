from .candidates import EntityCandidateExtractor
from .cleanup import EntitySpanCleaner
from .detector import EntityDetector
from .models import EntityResult
from .ranking import EntityRanker


class EntityEngine:
    """
    End-to-end entity intelligence engine.

    Pipeline:
        text
        -> candidates
        -> detection
        -> overlap cleanup
        -> ranking
        -> public EntityResult objects
    """

    def __init__(
        self,
        extractor: EntityCandidateExtractor | None = None,
        detector: EntityDetector | None = None,
        cleaner: EntitySpanCleaner | None = None,
        ranker: EntityRanker | None = None,
    ) -> None:

        self.extractor = (
            extractor
            or EntityCandidateExtractor()
        )

        self.detector = (
            detector
            or EntityDetector()
        )

        self.cleaner = (
            cleaner
            or EntitySpanCleaner()
        )

        self.ranker = (
            ranker
            or EntityRanker()
        )

    def analyze(
        self,
        text: str,
    ) -> list[EntityResult]:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        # 1. Candidate extraction
        candidates = self.extractor.extract(text)

        if not candidates:
            return []

        # 2. Entity detection
        detected = self.detector.detect(
            candidates,
            text,
        )

        if not detected:
            return []

        # 3. Remove overlapping spans
        cleaned = self.cleaner.clean(
            detected
        )

        if not cleaned:
            return []

        # 4. Rank entities
        ranked = self.ranker.rank(
            cleaned
        )

        # 5. Convert internal objects
        # into public result objects.
        return [
            self._to_result(entity)
            for entity in ranked
        ]

    @staticmethod
    def _to_result(entity) -> EntityResult:
        return EntityResult(
            text=entity.text,
            entity_type=entity.entity_type,
            score=entity.score,
            start=entity.start,
            end=entity.end,
        )