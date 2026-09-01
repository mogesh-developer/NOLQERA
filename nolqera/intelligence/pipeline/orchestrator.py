from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nolqera.intelligence.entities.engine import EntityEngine
from nolqera.intelligence.importance.engine import ImportanceEngine
from nolqera.intelligence.intent.engine import IntentEngine
from nolqera.intelligence.keyphrase.engine import KeyphraseEngine
from nolqera.intelligence.semantic_search.engine import SemanticSearchEngine
from nolqera.intelligence.semantic_search.models import (
    SemanticSearchResult,
)

from .context_compressor import ContextCompressor
from .context_ranker import ContextRankingAnalyzer
from .entity_analyzer import EntityAnalyzer
from .importance_analyzer import ImportanceAnalyzer
from .input_handler import InputHandler
from .intent_analyzer import IntentAnalyzer
from .keyword_analyzer import KeywordAnalyzer
from .noise_remover import NoiseRemover
from .relevance_analyzer import RelevanceAnalyzer
from .sentence_segmenter import SentenceSegmenter


@dataclass(frozen=True)
class PipelineResult:
    """
    Complete result produced by the NOLQERA core pipeline.
    """

    input_text: str
    normalized_text: str
    sentences: list[str]

    relevance: list[dict]
    importance: list[dict]

    keywords: Any
    entities: Any
    intents: Any

    filtered_results: list[SemanticSearchResult]
    ranked_context: list

    compressed_context: str

    @property
    def is_empty(self) -> bool:
        """
        Return True when the pipeline produced no
        optimized context.
        """
        return not bool(self.compressed_context)


class NOLQERAPipeline:
    """
    Core NOLQERA orchestration pipeline.

    This class does not implement new intelligence.

    It coordinates the existing Phase 1-4 intelligence
    modules through the pipeline adapters.

    Flow:

        raw input
            ↓
        input handling
            ↓
        sentence segmentation
            ↓
        relevance analysis
            ↓
        noise removal
            ↓
        importance analysis
            ↓
        keyword analysis
            ↓
        entity analysis
            ↓
        intent analysis
            ↓
        context ranking
            ↓
        context compression
            ↓
        PipelineResult
    """

    def __init__(
        self,
        semantic_search_engine: SemanticSearchEngine,
        importance_engine: ImportanceEngine,
        keyphrase_engine: KeyphraseEngine,
        entity_engine: EntityEngine,
        intent_engine: IntentEngine,
        noise_remover: NoiseRemover,
        context_ranker: ContextRankingAnalyzer,
        context_compressor: ContextCompressor,
        input_handler: InputHandler | None = None,
        sentence_segmenter: SentenceSegmenter | None = None,
        keyword_top_k: int = 5,
        max_sentences: int = 3,
    ) -> None:

        if not isinstance(
            semantic_search_engine,
            SemanticSearchEngine,
        ):
            raise TypeError(
                "semantic_search_engine must be "
                "a SemanticSearchEngine"
            )

        if not isinstance(
            importance_engine,
            ImportanceEngine,
        ):
            raise TypeError(
                "importance_engine must be "
                "an ImportanceEngine"
            )

        if not isinstance(
            keyphrase_engine,
            KeyphraseEngine,
        ):
            raise TypeError(
                "keyphrase_engine must be "
                "a KeyphraseEngine"
            )

        if not isinstance(
            entity_engine,
            EntityEngine,
        ):
            raise TypeError(
                "entity_engine must be "
                "an EntityEngine"
            )

        if not isinstance(
            intent_engine,
            IntentEngine,
        ):
            raise TypeError(
                "intent_engine must be "
                "an IntentEngine"
            )

        if not isinstance(
            noise_remover,
            NoiseRemover,
        ):
            raise TypeError(
                "noise_remover must be a NoiseRemover"
            )

        if not isinstance(
            context_ranker,
            ContextRankingAnalyzer,
        ):
            raise TypeError(
                "context_ranker must be "
                "a ContextRankingAnalyzer"
            )

        if not isinstance(
            context_compressor,
            ContextCompressor,
        ):
            raise TypeError(
                "context_compressor must be "
                "a ContextCompressor"
            )

        if input_handler is None:
            input_handler = InputHandler()

        if not isinstance(
            input_handler,
            InputHandler,
        ):
            raise TypeError(
                "input_handler must be an InputHandler"
            )

        if sentence_segmenter is None:
            sentence_segmenter = SentenceSegmenter()

        if not isinstance(
            sentence_segmenter,
            SentenceSegmenter,
        ):
            raise TypeError(
                "sentence_segmenter must be "
                "a SentenceSegmenter"
            )

        if not isinstance(keyword_top_k, int):
            raise TypeError(
                "keyword_top_k must be an integer"
            )

        if keyword_top_k <= 0:
            raise ValueError(
                "keyword_top_k must be greater than zero"
            )

        if not isinstance(max_sentences, int):
            raise TypeError(
                "max_sentences must be an integer"
            )

        if max_sentences <= 0:
            raise ValueError(
                "max_sentences must be greater than zero"
            )

        self.input_handler = input_handler
        self.sentence_segmenter = sentence_segmenter

        self.relevance_analyzer = RelevanceAnalyzer(
            semantic_search_engine
        )

        self.importance_analyzer = ImportanceAnalyzer(
            importance_engine
        )

        self.keyword_analyzer = KeywordAnalyzer(
            keyphrase_engine
        )

        self.entity_analyzer = EntityAnalyzer(
            entity_engine
        )

        self.intent_analyzer = IntentAnalyzer(
            intent_engine
        )

        self.noise_remover = noise_remover
        self.context_ranker = context_ranker
        self.context_compressor = context_compressor

        self.keyword_top_k = keyword_top_k
        self.max_sentences = max_sentences

    def process(
        self,
        query: str,
        raw_input: str,
    ) -> PipelineResult:
        """
        Execute the complete NOLQERA pipeline.

        Parameters
        ----------
        query:
            User query used for relevance analysis.

        raw_input:
            Raw context/document text to process.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not isinstance(raw_input, str):
            raise TypeError(
                "raw_input must be a string"
            )

        if not raw_input.strip():
            raise ValueError(
                "raw_input cannot be empty"
            )

        # ---------------------------------------------------------
        # 1. Input handling
        # ---------------------------------------------------------

        normalized_text = self.input_handler.handle(
            raw_input
        )

        # ---------------------------------------------------------
        # 2. Sentence segmentation
        # ---------------------------------------------------------

        sentences = self.sentence_segmenter.segment(
            normalized_text
        )

        # ---------------------------------------------------------
        # 3. Relevance analysis
        # ---------------------------------------------------------

        relevance = self.relevance_analyzer.analyze(
            query,
            sentences,
        )

        relevance_results = [
            SemanticSearchResult(
                text=item["text"],
                score=item["score"],
                index=item["index"],
            )
            for item in relevance
        ]

        # ---------------------------------------------------------
        # 4. Noise removal
        # ---------------------------------------------------------

        filtered_results = self.noise_remover.remove(
            relevance_results
        )

        filtered_indexes = {
            result.index
            for result in filtered_results
        }

        filtered_sentences = [
            sentence
            for index, sentence in enumerate(sentences)
            if index in filtered_indexes
        ]

        # ---------------------------------------------------------
        # 5. Importance analysis
        # ---------------------------------------------------------

        if filtered_sentences:
            importance = self.importance_analyzer.analyze(
                filtered_sentences
            )
        else:
            importance = []

        importance_by_text = {
            item["text"]: item["score"]
            for item in importance
        }

        importance_scores = [
            importance_by_text.get(
                result.text,
                0.0,
            )
            for result in filtered_results
        ]

        # ---------------------------------------------------------
        # 6. Keyword / keyphrase analysis
        # ---------------------------------------------------------

        keywords = self.keyword_analyzer.analyze(
            normalized_text,
            top_k=self.keyword_top_k,
        )

        # ---------------------------------------------------------
        # 7. Entity analysis
        # ---------------------------------------------------------

        entities = self.entity_analyzer.analyze(
            normalized_text
        )

        # ---------------------------------------------------------
        # 8. Intent analysis
        # ---------------------------------------------------------

        intents = self.intent_analyzer.analyze(
            normalized_text
        )

        # ---------------------------------------------------------
        # 9. Context ranking
        # ---------------------------------------------------------

        ranked_context = self.context_ranker.rank(
            filtered_results,
            importance_scores,
        )

        # ---------------------------------------------------------
        # 10. Context compression
        # ---------------------------------------------------------

        compressed_context = (
            self.context_compressor.compress(
                ranked_context,
                max_sentences=self.max_sentences,
            )
        )

        # ---------------------------------------------------------
        # 11. Unified result
        # ---------------------------------------------------------

        return PipelineResult(
            input_text=raw_input,
            normalized_text=normalized_text,
            sentences=sentences,
            relevance=relevance,
            importance=importance,
            keywords=keywords,
            entities=entities,
            intents=intents,
            filtered_results=filtered_results,
            ranked_context=ranked_context,
            compressed_context=compressed_context,
        )