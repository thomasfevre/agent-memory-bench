from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_context_shard_review_pack_is_balanced_and_source_backed():
    corpus_ids = {
        row["id"] for row in read_jsonl(ROOT / "data" / "corpus.jsonl")
    }
    candidates = read_jsonl(
        ROOT / "data" / "context-shard-review-candidates.jsonl"
    )

    assert len(candidates) == 24
    assert len({row["id"] for row in candidates}) == 24
    assert Counter(row["review"] for row in candidates) == {
        "approved": 8,
        "rejected": 8,
        "deferred": 8,
    }
    assert all(row["source_ids"] for row in candidates)
    assert all(
        set(row["source_ids"]).issubset(corpus_ids) for row in candidates
    )
