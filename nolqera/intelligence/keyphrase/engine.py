from ...features.tfidf import TfidfVectorizer
from ...tokenization.tokenizer import Tokenizer

from .candidates import KeyphraseCandidateExtractor
from .models import KeyphraseResult
from .ranking import KeyphraseRanker
from .scorer import KeyphraseScorer


class KeyphraseEngine:
    """Extract and rank meaningful keyphrases from text."""

    def __init__(
        self,
        tfidf_weight: float = 0.6,
        frequency_weight: float = 0.3,
        length_weight: float = 0.1,
        min_n: int = 1,
        max_n: int = 3,
    ):
        self.tokenizer = Tokenizer()

        self.candidate_extractor = (
            KeyphraseCandidateExtractor(
                min_n=min_n,
                max_n=max_n,
            )
        )

        self.scorer = KeyphraseScorer(
            tfidf_weight=tfidf_weight,
            frequency_weight=frequency_weight,
            length_weight=length_weight,
        )

        self.ranker = KeyphraseRanker()

    def extract(
        self,
        text: str,
        top_k: int | None = None,
    ) -> list[KeyphraseResult]:
        """Extract ranked keyphrases from a document."""

        # ---------------------------------------------------------
        # 1. Validate input
        # ---------------------------------------------------------

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        if top_k is not None:
            if not isinstance(top_k, int):
                raise TypeError(
                    "top_k must be an integer or None"
                )

            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero"
                )

        # ---------------------------------------------------------
        # 2. Tokenize document
        # ---------------------------------------------------------

        tokens = self.tokenizer.tokenize(text)

        if not tokens:
            raise ValueError(
                "Text must contain at least one token"
            )

        # Normalize tokens so candidate extraction and TF-IDF
        # use exactly the same representation.
        tokens = [
            token.strip().lower()
            for token in tokens
            if token.strip()
        ]

        if not tokens:
            raise ValueError(
                "Text must contain valid tokens"
            )

        # ---------------------------------------------------------
        # 3. Generate keyphrase candidates
        # ---------------------------------------------------------

        candidates = self.candidate_extractor.extract(
            tokens
        )

        if not candidates:
            return []

        # ---------------------------------------------------------
        # 4. Fit TF-IDF on the ACTUAL document
        # ---------------------------------------------------------

        vectorizer = TfidfVectorizer()

        document_vectors = vectorizer.fit_transform(
            [tokens]
        )

        document_vector = document_vectors[0]

        # Map:
        #
        # token -> TF-IDF score
        #
        token_tfidf: dict[str, float] = {}

        for index, value in enumerate(document_vector):
            token = vectorizer.vocabulary.get_token(index)

            token_tfidf[token] = value

        # ---------------------------------------------------------
        # 5. Calculate TF-IDF signal for every candidate
        # ---------------------------------------------------------

        tfidf_scores = []

        for candidate in candidates:
            candidate_tokens = candidate.split()

            values = [
                token_tfidf.get(token, 0.0)
                for token in candidate_tokens
            ]

            if not values:
                score = 0.0
            else:
                score = sum(values) / len(values)

            tfidf_scores.append(score)

        # Normalize TF-IDF scores to 0-1.
        max_tfidf = max(tfidf_scores)

        if max_tfidf == 0:
            normalized_tfidf = [
                0.0
                for _ in tfidf_scores
            ]
        else:
            normalized_tfidf = [
                score / max_tfidf
                for score in tfidf_scores
            ]

        # ---------------------------------------------------------
        # 6. Calculate frequency signal
        # ---------------------------------------------------------

        total_tokens = len(tokens)

        frequencies = []

        for candidate in candidates:
            candidate_tokens = candidate.split()

            count = 0

            window_count = (
                total_tokens
                - len(candidate_tokens)
                + 1
            )

            for index in range(
                max(window_count, 0)
            ):
                window = tokens[
                    index:index + len(candidate_tokens)
                ]

                if window == candidate_tokens:
                    count += 1

            frequencies.append(count)

        # Normalize frequency scores to 0-1.
        max_frequency = max(frequencies)

        if max_frequency == 0:
            normalized_frequency = [
                0.0
                for _ in frequencies
            ]
        else:
            normalized_frequency = [
                frequency / max_frequency
                for frequency in frequencies
            ]

        # ---------------------------------------------------------
        # 7. Calculate phrase-length signal
        # ---------------------------------------------------------

        normalized_length = []

        for candidate in candidates:
            phrase_length = len(candidate.split())

            # 1 word  -> 0.33
            # 2 words -> 0.67
            # 3+ words -> 1.00
            length_score = min(
                phrase_length / 3.0,
                1.0,
            )

            normalized_length.append(
                length_score
            )

        # ---------------------------------------------------------
        # 8. Combine all signals
        # ---------------------------------------------------------

        scores: dict[str, float] = {}

        for (
            candidate,
            tfidf,
            frequency,
            length,
        ) in zip(
            candidates,
            normalized_tfidf,
            normalized_frequency,
            normalized_length,
        ):
            scores[candidate] = self.scorer.score(
                tfidf_score=tfidf,
                frequency_score=frequency,
                length_score=length,
            )

        # ---------------------------------------------------------
        # 9. Rank candidates
        # ---------------------------------------------------------

        ranked = self.ranker.rank(scores)

        # Remove shorter overlapping phrases before top_k
        ranked = self.ranker.remove_overlapping(ranked)

        # ---------------------------------------------------------
        # 10. Apply top_k
        # ---------------------------------------------------------

        if top_k is not None:
            ranked = ranked[:top_k]

        # ---------------------------------------------------------
        # 11. Convert to public result model
        # ---------------------------------------------------------

        return [
            KeyphraseResult(
                phrase=item.phrase,
                score=item.score,
                rank=index,
            )
            for index, item in enumerate(
                ranked,
                start=1,
            )
        ]