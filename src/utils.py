from math import ceil
from typing import Literal


QueryBucket = Literal["short", "medium", "long"]


def token_count(text: str) -> int:
    return len(text.split())


def query_length_bucket(query: str) -> QueryBucket:
    n_tokens = token_count(query)
    if n_tokens < 5:
        return "short"
    if n_tokens <= 15:
        return "medium"
    return "long"


def top_10_percent_count(total_docs: int) -> int:
    return ceil(total_docs * 0.10)