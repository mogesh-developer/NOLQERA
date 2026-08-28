from ...features.tfidf import TfidfVectorizer
from ...preprocessing.pipeline import PreprocessingPipeline
from .models import RelevanceResult
from .ranking import RelevanceRanker
from .scorer import RelevanceScorer
from .similarity import cosine_similarity


class RelevanceEngine:
    """Analyze and rank sentence relevance against a query."""

    def __init__(
        self,
        relevant_threshold: float = 0.50,
        weak_threshold: float = 0.20,
        pipeline: PreprocessingPipeline | None = None,
    ):
        self.scorer = RelevanceScorer(
            relevant_threshold=relevant_threshold,
            weak_threshold=weak_threshold,
        )

        self.ranker = RelevanceRanker()

        if pipeline is None:
            pipeline = PreprocessingPipeline(
                remove_stopwords=True,
                stemming=True,
            )
        self.pipeline = pipeline

    def analyze(
        self,
        query: str,
        sentences: list[str],
    ) -> list[RelevanceResult]:
        """Calculate and rank sentence relevance."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list")

        if not query.strip():
            raise ValueError("query cannot be empty")

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

        query_tokens = self.pipeline.process(query).split()

        sentence_tokens = [
            self.pipeline.process(sentence).split()
            for sentence in sentences
        ]

        documents = [
            query_tokens,
            *sentence_tokens,
        ]

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(documents)

        query_vector = vectors[0]
        sentence_vectors = vectors[1:]

        similarities = [
            cosine_similarity(
                query_vector,
                sentence_vector,
            )
            for sentence_vector in sentence_vectors
        ]

        scored = [
            self.scorer.score(similarity)
            for similarity in similarities
        ]

        ranked = self.ranker.rank(
            [result.score for result in scored]
        )

        results = []

        for rank, item in enumerate(ranked, start=1):
            index = item.index
            score = scored[index]

            results.append(
                RelevanceResult(
                    sentence=sentences[index],
                    score=score.score,
                    label=score.label,
                    rank=rank,
                )
            )

        return results