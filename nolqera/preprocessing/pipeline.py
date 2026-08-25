from ..tokenization.word_tokenizer import tokenize_words

from .cleaner import (
    remove_email_addresses,
    remove_extra_whitespace,
    remove_urls,
)
from .lemmatization import SimpleLemmatizer
from .normalizer import normalize
from .stemming import SimpleStemmer
from .stopwords import StopwordRemover


class PreprocessingPipeline:
    """Configurable text preprocessing pipeline."""

    def __init__(
        self,
        remove_stopwords: bool = False,
        stopwords: set[str] | None = None,
        stemming: bool = False,
        lemmatization: bool = False,
    ):
        if stemming and lemmatization:
            raise ValueError(
                "stemming and lemmatization "
                "cannot both be enabled"
            )

        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization

        self.stopword_remover = (
            StopwordRemover(stopwords)
            if remove_stopwords
            else None
        )

        self.stemmer = (
            SimpleStemmer()
            if stemming
            else None
        )

        self.lemmatizer = (
            SimpleLemmatizer()
            if lemmatization
            else None
        )

    def process(self, text: str) -> str:
        """Preprocess text."""

        text = remove_urls(text)
        text = remove_email_addresses(text)
        text = remove_extra_whitespace(text)
        text = normalize(text)

        tokens = tokenize_words(text)

        if self.remove_stopwords:
            tokens = self.stopword_remover.remove(tokens)

        if self.stemming:
            tokens = self.stemmer.stem_tokens(tokens)

        elif self.lemmatization:
            tokens = self.lemmatizer.lemmatize_tokens(tokens)

        return " ".join(tokens)


def preprocess(text: str) -> str:
    """Run the default preprocessing pipeline."""

    pipeline = PreprocessingPipeline()

    return pipeline.process(text)