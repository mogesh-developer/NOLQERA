from nolqera.features.tfidf import TfidfVectorizer
from nolqera.preprocessing.pipeline import PreprocessingPipeline
from nolqera.relevance.ranking import RelevanceRanker
from nolqera.relevance.scorer import RelevanceScorer
from nolqera.relevance.similarity import cosine_similarity


def test_relevance_pipeline_ranks_real_context():
    query = "What database does the application use?"

    sentences = [
        "The application is built using FastAPI.",
        "The application uses MongoDB for data storage.",
        "I travelled to Chennai yesterday.",
        "The API provides REST endpoints.",
    ]

    pipeline = PreprocessingPipeline(
        remove_stopwords=True,
        stemming=True,
    )

    query_tokens = pipeline.process(query).split()

    sentence_tokens = [
        pipeline.process(sentence).split()
        for sentence in sentences
    ]

    documents = [query_tokens, *sentence_tokens]

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

    scorer = RelevanceScorer(
        relevant_threshold=0.05,
        weak_threshold=0.01,
    )

    scored = [
        scorer.score(similarity)
        for similarity in similarities
    ]

    ranker = RelevanceRanker()

    ranked = ranker.rank(
        [result.score for result in scored]
    )

    print("\n--- NOLQERA Relevance Pipeline ---")

    for item in ranked:
        index = item.index
        result = scored[index]

        print(
            f"{item.score:.4f} | "
            f"{result.label:10} | "
            f"{sentences[index]}"
        )

    best_index = ranked[0].index

    print("\n--- Selected Context ---")
    print(sentences[best_index])

    assert (
        sentences[best_index]
        == "The application uses MongoDB for data storage."
    )

    assert scored[best_index].label == "relevant"