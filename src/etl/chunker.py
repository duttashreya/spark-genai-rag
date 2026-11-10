from typing import List, Dict, Any
from pyspark.sql.functions import udf, explode, col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Splits text into chunks of roughly `chunk_size` characters with `chunk_overlap`.
    Returns a list of dicts with chunk_index and content.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    start_idx = 0
    chunk_idx = 0

    while start_idx < len(words):
        # Slice window of words
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_content = " ".join(chunk_words)

        chunks.append({
            "chunk_index": chunk_idx,
            "content": chunk_content,
        })

        if end_idx == len(words):
            break

        # Advance window by chunk_size - chunk_overlap
        step = max(1, chunk_size - chunk_overlap)
        start_idx += step
        chunk_idx += 1

    return chunks


chunk_schema = ArrayType(
    StructType([
        StructField("chunk_index", IntegerType(), False),
        StructField("content", StringType(), False),
    ])
)


@udf(returnType=chunk_schema)
def chunk_text_udf(text):
    if text is None:
        return []
    return chunk_text(text, chunk_size=100, chunk_overlap=20)