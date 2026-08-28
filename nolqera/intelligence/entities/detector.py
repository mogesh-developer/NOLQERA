from dataclasses import dataclass

from .candidates import EntityCandidate


@dataclass(frozen=True)
class DetectedEntity:
    """Entity candidate with an inferred type and confidence."""

    candidate: EntityCandidate
    entity_type: str
    score: float

    @property
    def text(self) -> str:
        return self.candidate.text

    @property
    def start(self) -> int:
        return self.candidate.start

    @property
    def end(self) -> int:
        return self.candidate.end


class EntityDetector:
    """Infer entity types from linguistic characteristics."""

    _LOCATION_CONTEXT = {
        "in",
        "at",
        "from",
        "near",
        "around",
        "to",
    }

    _ORGANIZATION_CONTEXT = {
        "company",
        "organization",
        "corporation",
        "university",
        "college",
        "school",
        "institute",
        "academy",
    }

    _PERSON_CONTEXT = {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "by",
    }

    def detect(
        self,
        candidates: list[EntityCandidate],
        text: str,
    ) -> list[DetectedEntity]:
        """Infer entity types for candidate spans."""

        if not isinstance(candidates, list):
            raise TypeError(
                "candidates must be a list"
            )

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        results = []

        for candidate in candidates:
            if not isinstance(
                candidate,
                EntityCandidate,
            ):
                raise TypeError(
                    "all candidates must be "
                    "EntityCandidate instances"
                )

            entity_type, score = self._classify(
                candidate,
                text,
            )

            results.append(
                DetectedEntity(
                    candidate=candidate,
                    entity_type=entity_type,
                    score=score,
                )
            )

        return results

    def _classify(
        self,
        candidate: EntityCandidate,
        text: str,
    ) -> tuple[str, float]:
        """Infer entity type using contextual signals."""

        before = text[:candidate.start]

        context_tokens = (
            before.lower()
            .split()
        )

        previous_word = (
            context_tokens[-1].strip(".,!?;:")
            if context_tokens
            else ""
        )

        candidate_text = candidate.text

        # ---------------------------------------------------------
        # Acronym / uppercase signal
        # ---------------------------------------------------------

        if (
            len(candidate_text) >= 2
            and candidate_text.isupper()
            and candidate_text.isalpha()
        ):
            return "ORGANIZATION", 0.70

        # ---------------------------------------------------------
        # Explicit person-title context
        # ---------------------------------------------------------

        if previous_word in self._PERSON_CONTEXT:
            return "PERSON", 0.85

        # ---------------------------------------------------------
        # Organization-style contextual signal
        # ---------------------------------------------------------

        candidate_lower = (
            candidate_text.lower()
        )

        if any(
            word in candidate_lower.split()
            for word in self._ORGANIZATION_CONTEXT
        ):
            return "ORGANIZATION", 0.85

        # ---------------------------------------------------------
        # Location-style contextual signal
        # ---------------------------------------------------------

        if previous_word in self._LOCATION_CONTEXT:
            return "LOCATION", 0.65

        # ---------------------------------------------------------
        # Multi-word capitalized span
        # ---------------------------------------------------------

        if len(candidate_text.split()) > 1:
            return "ORGANIZATION", 0.55

        # ---------------------------------------------------------
        # Generic capitalized entity
        # ---------------------------------------------------------

        if candidate_text[:1].isupper():
            return "UNKNOWN", 0.40

        return "UNKNOWN", 0.20