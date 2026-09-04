import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityCandidate:
    """Represent a possible entity span."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")

        if not self.text.strip():
            raise ValueError("text cannot be empty")

        if not isinstance(self.start, int):
            raise TypeError("start must be an integer")

        if not isinstance(self.end, int):
            raise TypeError("end must be an integer")

        if self.start < 0:
            raise ValueError(
                "start cannot be negative"
            )

        if self.end <= self.start:
            raise ValueError(
                "end must be greater than start"
            )

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


class EntityCandidateExtractor:
    """Generate possible entity spans without entity dictionaries."""

    # Words that commonly connect words inside a proper noun.
    # These are linguistic connectors, NOT entity names.
    _COMMON_WORDS = {
        "the", "a", "an", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "we", "us", "our",
        "he", "him", "his", "she", "her", "hers", "you", "your",
        "yours", "i", "me", "my", "mine", "who", "whom", "whose",
        "which", "what", "where", "when", "why", "how", "if",
        "then", "else", "and", "or", "but", "so", "because",
        "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here",
        "there", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "than", "too", "very", "can",
        "will", "just", "should", "now", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "application", "system", "data"
    }

    def extract(
        self,
        text: str,
    ) -> list[EntityCandidate]:
        """Extract possible entity spans from text."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        candidates: list[EntityCandidate] = []

        # ---------------------------------------------------------
        # 1. Single capitalized words
        # ---------------------------------------------------------

        single_cap_pattern = re.compile(
            r"\b[A-Z][A-Za-z0-9]*\b"
        )

        for match in single_cap_pattern.finditer(text):
            value = match.group()

            if self._is_sentence_initial_word(
                text,
                match.start(),
                value,
            ) and value.lower() in self._COMMON_WORDS:
                continue

            candidates.append(
                EntityCandidate(
                    text=value,
                    start=match.start(),
                    end=match.end(),
                )
            )

        # ---------------------------------------------------------
        # 2. Multi-word capitalized phrases
        # ---------------------------------------------------------

        multi_cap_pattern = re.compile(
            r"\b[A-Z][A-Za-z0-9]*(?:"
            r"\s+(?:"
            r"[A-Z][A-Za-z0-9]*"
            r"|"
            r"(?:of|the|and|for|at|in)"
            r"\s+[A-Z][A-Za-z0-9]*"
            r")"
            r")+\b"
        )

        for match in multi_cap_pattern.finditer(text):
            candidates.append(
                EntityCandidate(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )

        # ---------------------------------------------------------
        # 3. Acronym / uppercase candidates
        # ---------------------------------------------------------

        acronym_pattern = re.compile(
            r"\b[A-Z]{2,}(?:[A-Z0-9-]*[A-Z0-9])?\b"
        )

        for match in acronym_pattern.finditer(text):
            candidates.append(
                EntityCandidate(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )

        # ---------------------------------------------------------
        # 4. Mixed alphanumeric / technical-looking spans
        # ---------------------------------------------------------

        mixed_pattern = re.compile(
            r"\b[A-Za-z]+[A-Z][A-Za-z0-9]*\b"
            r"|"
            r"\b[A-Za-z]+[0-9][A-Za-z0-9-]*\b"
            r"|"
            r"\b[A-Z][A-Za-z]+[0-9][A-Za-z0-9-]*\b"
        )

        for match in mixed_pattern.finditer(text):
            candidates.append(
                EntityCandidate(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )

        # ---------------------------------------------------------
        # 4. Remove exact duplicate spans
        # ---------------------------------------------------------

        unique: dict[
            tuple[int, int],
            EntityCandidate,
        ] = {}

        for candidate in candidates:
            key = (
                candidate.start,
                candidate.end,
            )

            unique[key] = candidate

        # ---------------------------------------------------------
        # 5. Sort by document position
        # ---------------------------------------------------------

        return sorted(
            unique.values(),
            key=lambda candidate: (
                candidate.start,
                candidate.end,
            ),
        )

    @staticmethod
    def _is_sentence_initial_word(
        text: str,
        start: int,
        value: str,
    ) -> bool:
        """Check whether a capitalized word is only sentence-initial."""

        if len(value.split()) != 1:
            return False

        if start == 0:
            return True

        previous = text[:start].rstrip()

        return previous.endswith(
            (".", "!", "?")
        )