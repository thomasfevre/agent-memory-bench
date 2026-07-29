from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


SOURCE_PATTERN = re.compile(r"\b(?:SOURCE\s+)?(d\d{2})\b", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        return flatten_text(value.model_dump())
    if isinstance(value, dict):
        return "\n".join(flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return "\n".join(flatten_text(item) for item in value)
    return str(value)


def source_ids_from_text(value: Any) -> list[str]:
    seen: list[str] = []
    for match in SOURCE_PATTERN.findall(flatten_text(value)):
        source_id = match.lower()
        if source_id not in seen:
            seen.append(source_id)
    return seen


def score_retrieval(
    questions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {row["question_id"]: row for row in rows}
    recalls: list[float] = []
    precisions: list[float] = []
    temporal_correct: list[float] = []
    temporal_precisions: list[float] = []
    temporal_exact_sets: list[float] = []
    abstentions: list[float] = []
    latencies = [float(row["latency_ms"]) for row in rows]

    for question in questions:
        row = by_id[question["id"]]
        gold = set(question["gold_source_ids"])
        retrieved = set(row["retrieved_source_ids"])
        if gold:
            recalls.append(len(gold & retrieved) / len(gold))
            precisions.append(len(gold & retrieved) / max(len(retrieved), 1))
        else:
            abstentions.append(1.0 if not retrieved else 0.0)
        if "temporal" in question["category"]:
            temporal_correct.append(1.0 if gold and gold.issubset(retrieved) else 0.0)
            temporal_precisions.append(len(gold & retrieved) / max(len(retrieved), 1))
            temporal_exact_sets.append(1.0 if gold == retrieved else 0.0)

    return {
        "questions": len(questions),
        "mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "mean_context_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "temporal_correctness": (
            sum(temporal_correct) / len(temporal_correct) if temporal_correct else 0.0
        ),
        "temporal_context_precision": (
            sum(temporal_precisions) / len(temporal_precisions) if temporal_precisions else 0.0
        ),
        "temporal_exact_source_set": (
            sum(temporal_exact_sets) / len(temporal_exact_sets) if temporal_exact_sets else 0.0
        ),
        "abstention_accuracy": sum(abstentions) / len(abstentions) if abstentions else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": (
            sorted(latencies)[min(math.ceil(len(latencies) * 0.95) - 1, len(latencies) - 1)]
            if latencies
            else 0.0
        ),
    }


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
