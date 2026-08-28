from ...features.tfidf import TfidfVectorizer
from ...preprocessing.pipeline import PreprocessingPipeline
from .models import ImportanceResult
from .ranking import ImportanceRanker
from .scorer import ImportanceScorer


class ImportanceEngine:
    """Analyze and rank sentence importance within a document."""

    def __init__(
        self,
        tfidf_weight: float = 0.7,
        position_weight: float = 0.1,
        density_weight: float = 0.2,
        pipeline: PreprocessingPipeline | None = None,
    ):
        self.scorer = ImportanceScorer(
            tfidf_weight=tfidf_weight,
            position_weight=position_weight,
            density_weight=density_weight,
        )

        self.ranker = ImportanceRanker()

        if pipeline is None:
            pipeline = PreprocessingPipeline(
                remove_stopwords=True,
                stemming=True,
            )
        self.pipeline = pipeline

    def analyze(
        self,
        sentences: list[str],
    ) -> list[ImportanceResult]:
        """Calculate importance for each sentence."""

        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list")

        if not sentences:
            raise ValueError("sentences cannot be empty")

        if any(
            not isinstance(sentence, str)
            for sentence in sentences
        ):
            raise TypeError(
                "sentences must contain only strings"
            )

        if any(
            not sentence.strip()
            for sentence in sentences
        ):
            raise ValueError(
                "sentences cannot contain empty strings"
            )

        tokenized_sentences = [
            self.pipeline.process(sentence).split()
            for sentence in sentences
        ]

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            tokenized_sentences
        )

        # Sentence-level TF-IDF signal.
        # Use the average of non-zero TF-IDF values
        # instead of allowing one token to dominate.
        tfidf_scores = []

        for vector in vectors:
            non_zero_values = [
                value for value in vector
                if value > 0
            ]

            if not non_zero_values:
                tfidf_scores.append(0.0)
            else:
                tfidf_scores.append(sum(non_zero_values))

        total_sentences = len(sentences)

        max_tfidf = max(tfidf_scores)

        if max_tfidf == 0:
            if total_sentences == 1:
                normalized_tfidf = [1.0]
            else:
                normalized_tfidf = [
                    0.0 for _ in tfidf_scores
                ]
        else:
            normalized_tfidf = [
                score / max_tfidf
                for score in tfidf_scores
            ]

        position_scores = [
            1.0 - (
                index / total_sentences
            )
            for index in range(total_sentences)
        ]

        density_scores = []

        for tokens in tokenized_sentences:
            if not tokens:
                density_scores.append(0.0)
                continue

            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)

            density_scores.append(
                unique_tokens / total_tokens
            )

        scores = []

        for tfidf, position, density in zip(
            normalized_tfidf,
            position_scores,
            density_scores,
        ):
            scores.append(
                self.scorer.score(
                    tfidf_score=tfidf,
                    position_score=position,
                    density_score=density,
                )
            )

        ranked = self.ranker.rank(scores)

        return [
            ImportanceResult(
                sentence=sentences[item.index],
                score=item.score,
                rank=item.rank,
            )
            for item in ranked
        ]