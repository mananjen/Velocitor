from src.utils import query_length_bucket, token_count, top_10_percent_count


def test_token_count_splits_on_whitespace() -> None:
    assert token_count("what is short selling") == 4
    assert token_count("  one   two   three  ") == 3
    assert token_count("") == 0


def test_query_length_bucket_boundaries() -> None:
    assert query_length_bucket("one two three four") == "short"
    assert query_length_bucket("one two three four five") == "medium"
    assert query_length_bucket(" ".join(["x"] * 15)) == "medium"
    assert query_length_bucket(" ".join(["x"] * 16)) == "long"


def test_top_10_percent_count_rounds_up() -> None:
    assert top_10_percent_count(10) == 1
    assert top_10_percent_count(11) == 2
    assert top_10_percent_count(99) == 10
    assert top_10_percent_count(57638) == 5764