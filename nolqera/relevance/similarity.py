import math

def cosine_similarity(
    vector_a : list[float],
    vector_b : list[float],
) -> float:
    """Calculate cosinr similarity b/w the 2 num numbers"""

    if not isinstance(vector_a, list):
        raise TypeError("vector_a ah list ah venum da")
    
    if not isinstance(vector_b, list):
        raise TypeError("vectot_b ah list ah venum da")
    
    if not vector_a or not vector_b:
        raise ValueError("Vector empty ah irrukka koodathu da")

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension")

    if any( 
        not isinstance(value, (int, float))
        for value in vector_a
    ):
        raise TypeError("vector_a numbers ah irukkanum da")
    
    if any(
        not isinstance(value, (int, float))
        for value in vector_b
    ):
        raise TypeError("vector_b numbers ah irukkanum da")
    
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )
    
    