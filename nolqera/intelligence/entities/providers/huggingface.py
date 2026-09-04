from transformers import pipeline


class HuggingFaceEntityRecognizer:
    """
    Hugging Face based entity recognizer.

    Uses a pretrained token-classification model and normalizes
    its output into NOLQERA-friendly entity dictionaries.
    """

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name

        self._pipeline = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
        )

    def recognize(self, text: str) -> list[dict]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text cannot be empty")

        entities = self._pipeline(text)

        return [
            {
                "text": entity["word"],
                "entity_type": entity["entity_group"],
                "score": float(entity["score"]),
                "start": int(entity["start"]),
                "end": int(entity["end"]),
            }
            for entity in entities
        ]