#!/usr/bin/env python3
"""Refresh the common-corpus public records from PROTOTYPE-latest.json."""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "PROTOTYPE-latest.json"
REGISTRY = ROOT / "results" / "published" / "registry.json"

RETRIEVAL_LABELS = {
    "long_context": "Long context",
    "bm25": "BM25",
    "dense": "MiniLM dense",
    "hybrid": "Hybrid",
    "facts": "Dated facts",
    "graph": "Dated graph",
    "routed": "Type router",
    "parallel_merge": "Parallel fusion",
}
GENERATION_LABELS = {
    "long_context": "Long context",
    "hybrid": "Hybrid",
    "routed": "Type router",
    "parallel_merge": "Parallel fusion",
}


def rounded(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def by_id(registry: dict[str, Any], run_id: str) -> dict[str, Any]:
    return next(run for run in registry["runs"] if run["id"] == run_id)


def main() -> int:
    if not SOURCE.is_file():
        print(f"missing {SOURCE.relative_to(ROOT)}")
        return 1

    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    created_at = result["manifest"]["created_at"]
    run_date = datetime.fromisoformat(created_at).date().isoformat()
    registry["updated_at"] = run_date

    rows_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in result["retrieval"]["rows"]:
        rows_by_strategy.setdefault(row["strategy"], []).append(row)

    existing_retrieval = by_id(registry, "common-retrieval-20260729")
    previous_words = {
        item["method"]: item.get("mean_context_words")
        for item in existing_retrieval["metrics"]["series"]
    }
    retrieval_series = []
    for summary in result["retrieval"]["summaries"]:
        strategy = summary["strategy"]
        if strategy not in RETRIEVAL_LABELS:
            continue
        label = RETRIEVAL_LABELS[strategy]
        context_words = [
            row["context_words"]
            for row in rows_by_strategy.get(strategy, [])
            if row.get("context_words") is not None
        ]
        retrieval_series.append(
            {
                "method": label,
                "recall": rounded(summary["recall"]),
                "context_precision": rounded(summary["context_precision"]),
                "temporal_correctness": rounded(summary["temporal_exact"]),
                "latency_ms": rounded(summary["latency_ms"], 2),
                "mean_context_words": (
                    rounded(statistics.fmean(context_words), 1)
                    if context_words
                    else previous_words.get(label)
                ),
            }
        )
    existing_retrieval["date"] = run_date
    existing_retrieval["repetitions"] = result["manifest"]["repetitions"]
    existing_retrieval["metrics"]["series"] = retrieval_series

    ollama = result.get("ollama")
    if ollama and ollama.get("summaries"):
        generation = by_id(registry, "common-generation-qwen8-20260729")
        reader_model = ollama["summaries"][0]["model"]
        generation["date"] = run_date
        generation["reader"] = reader_model
        generation["repetitions"] = result["manifest"]["repetitions"]
        generation["metrics"]["series"] = [
            {
                "method": GENERATION_LABELS[item["strategy"]],
                "accuracy": rounded(item["accuracy"]),
                "tokens": item["prompt_tokens"] + item["output_tokens"],
                "latency_s": rounded(item["latency_ms"] / 1000, 2),
            }
            for item in ollama["summaries"]
            if item["strategy"] in GENERATION_LABELS
        ]

    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(REGISTRY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
