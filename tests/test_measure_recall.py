# tests/test_measure_recall.py — C6 + generator handling.
from scripts.measure_recall import recall_at_10


def test_duplicate_prediction_query_id_is_an_error():
    gt = [{"query_id": 0, "top10": ["a"]}]
    pred = [
        {"query_id": 0, "top10": ["a"]},
        {"query_id": 0, "top10": ["b"]},
    ]
    try:
        recall_at_10(gt, pred)
    except ValueError as e:
        assert "duplicate query_id" in str(e)
    else:  # pragma: no cover
        raise AssertionError("duplicate query_id was silently accepted")


def test_prediction_generator_is_not_exhausted_by_the_duplicate_check():
    """pred_rows may be a generator (the docstring says iterable).

    The old two-pass implementation consumed it in the duplicate check, so
    every gt row looked unpredicted (recall None, everything skipped).
    """
    gt = [{"query_id": 0, "top10": ["a", "b"]}]
    pred = ({"query_id": 0, "top10": ["a", "b"]} for _ in range(1))
    recall, skipped = recall_at_10(gt, pred)
    assert recall == 0.2  # 2 of the 10 window ids match
    assert skipped == 0


def test_unpredicted_gt_rows_are_skipped_and_counted():
    gt = [
        {"query_id": 0, "top10": ["a"]},
        {"query_id": 1, "top10": ["b"]},
    ]
    pred = [{"query_id": 0, "top10": ["a"]}]
    seen = []
    recall, skipped = recall_at_10(gt, pred, on_skip=seen.append)
    assert recall == 0.1 and skipped == 1 and seen == [1]
