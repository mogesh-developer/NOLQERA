from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)

text = (
    "The application is built using FastAPI. "
    "The application uses MongoDB for data storage. "
    "Python is the main programming language."
)

results = ner(text)

for entity in results:
    print(
        entity["word"],
        "->",
        entity["entity_group"],
        "score=",
        round(entity["score"], 3),
    )