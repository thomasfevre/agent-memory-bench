import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memgym_dr_retrieval import deterministic_sample, load_jsonl, summarize_rows


def test_deterministic_sample_is_stable_and_source_ordered():
    rows = [{"instance_id": str(index)} for index in range(20)]
    first = deterministic_sample(rows, 5, 42)
    second = deterministic_sample(rows, 5, 42)
    assert first == second
    assert [int(row["instance_id"]) for row in first] == sorted(
        int(row["instance_id"]) for row in first
    )


def test_sample_zero_means_all_rows():
    rows = [{"instance_id": str(index)} for index in range(3)]
    assert deterministic_sample(rows, 0, 1) == rows


def test_load_jsonl_ignores_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value": 1}\n\n{"value": 2}\n')
    assert load_jsonl(path) == [{"value": 1}, {"value": 2}]


def test_summarize_rows():
    result = summarize_rows(
        [
            {
                "fact_recall": 1.0,
                "context_words": 100,
                "ingestion_seconds": 2.0,
                "query_ms": 4.0,
            },
            {
                "fact_recall": 0.0,
                "context_words": 200,
                "ingestion_seconds": 4.0,
                "query_ms": 8.0,
            },
        ]
    )
    assert result == {
        "n": 2,
        "mean_fact_recall": 0.5,
        "mean_context_words": 150.0,
        "mean_ingestion_seconds": 3.0,
        "mean_query_ms": 6.0,
    }
