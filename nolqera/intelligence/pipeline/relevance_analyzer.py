from nolqera.intelligence.semantic_search.engine import SemanticSearchEngine


class RelevanceAnalyzer:
    """
    Phase 4 adapter for the existing SemanticSearchEngine.

    Reuses the semantic-search implementation instead of
    calculating relevance independently.
    """

    def __init__(self, engine: SemanticSearchEngine):
        if not isinstance(engine, SemanticSearchEngine):
            raise TypeError(
                "engine must be a SemanticSearchEngine"
            )

        self.engine = engine

    def analyze(
        self,
        query: str,
        sentences: list[str],
    ) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list")

        if any(not isinstance(sentence, str) for sentence in sentences):
            raise TypeError(
                "sentences must contain only strings"
            )

        if any(not sentence.strip() for sentence in sentences):
            raise ValueError(
                "sentences cannot contain empty strings"
            )

        if not sentences:
            return []

        results = self.engine.search(
            query,
            documents=sentences,
            top_k=len(sentences),
        )

        scores = {
            result.index: result.score
            for result in results
        }

        return [
            {
                "index": index,
                "text": sentence,
                "score": scores[index],
            }
            for index, sentence in enumerate(sentences)
        ]