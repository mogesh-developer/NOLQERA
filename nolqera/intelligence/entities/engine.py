from .adapters import EntityAdapter
from .candidates import EntityCandidateExtractor
from .cleanup import EntitySpanCleaner
from .detector import EntityDetector
from .models import EntityResult
from .providers import HuggingFaceEntityRecognizer
from .ranking import EntityRanker


class EntityEngine:
    def __init__(
        self,
        extractor: EntityCandidateExtractor | None = None,
        detector: EntityDetector | None = None,
        cleaner: EntitySpanCleaner | None = None,
        ranker: EntityRanker | None = None,
        recognizer: HuggingFaceEntityRecognizer | None = None,
        adapter: EntityAdapter | None = None,
        use_external_recognizer: bool = False,
    ) -> None:
        self.extractor = extractor or EntityCandidateExtractor()
        self.detector = detector or EntityDetector()
        self.cleaner = cleaner or EntitySpanCleaner()
        self.ranker = ranker or EntityRanker()

        # Optional external NER provider.
        # Model is loaded only when explicitly provided.
        self.recognizer = recognizer
        self.adapter = adapter or EntityAdapter()

        if not isinstance(use_external_recognizer, bool):
            raise TypeError("use_external_recognizer must be a boolean")

        self.use_external_recognizer = use_external_recognizer

    def analyze(self, text: str) -> list[EntityResult]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        # Optional external NER path.
        # Existing heuristic pipeline remains the default.
        if self.use_external_recognizer:
            if self.recognizer is None:
                raise RuntimeError(
                    "external entity recognizer is enabled "
                    "but no recognizer is configured"
                )

            return self.analyze_with_recognizer(text)

        # Existing NOLQERA entity pipeline.
        candidates = self.extractor.extract(text)

        if not candidates:
            return []

        detected = self.detector.detect(candidates, text)

        if not detected:
            return []

        cleaned = self.cleaner.clean(detected)

        if not cleaned:
            return []

        ranked = self.ranker.rank(cleaned)

        return [self._to_result(entity) for entity in ranked]

    def analyze_with_recognizer(self, text: str) -> list[EntityResult]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        if self.recognizer is None:
            raise RuntimeError("entity recognizer is not configured")

        raw_entities = self.recognizer.recognize(text)

        return self.adapter.to_results(raw_entities)

    @staticmethod
    def _to_result(entity) -> EntityResult:
        return EntityResult(
            text=entity.text,
            entity_type=entity.entity_type,
            score=entity.score,
            start=entity.start,
            end=entity.end,
        )