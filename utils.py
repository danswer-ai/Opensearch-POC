from sentence_transformers import SentenceTransformer  # type:ignore
from sentence_transformers.util import cos_sim  # type:ignore
from enum import Enum


EMBEDDING_DIM = 384


class TextType(Enum):
    PASSAGE = "passage"
    QUERY = "query"

model = SentenceTransformer('intfloat/e5-small-v2')


def vectorize(text: str, text_type: TextType) -> list[float]:
    prefixed_text = f"{text_type.value}: {text}"
    embedding = model.encode(prefixed_text)
    return embedding.tolist()  # type:ignore


def get_cosine_sim(
    text1: str,
    text2: str
) -> float:
    embedding1 = vectorize(text1, TextType.PASSAGE)
    embedding2 = vectorize(text2, TextType.PASSAGE)
    similarity = cos_sim(embedding1, embedding2)
    return similarity.item()


def min_max_normalize(number: float, min_value: float, max_value: float) -> float:
    if min_value == max_value:
        raise ValueError("min_value and max_value must be different")
    return (number - min_value) / (max_value - min_value)
