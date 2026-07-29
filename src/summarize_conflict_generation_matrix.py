#!/usr/bin/env python3
"""Aggregate Conflict Resolution generation and evidence-retrieval outcomes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph_benchmark_common import write_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--generation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def merge_generation_rows(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    merged: dict[tuple[str, int, str], dict[str, Any]] = {}
    expected_questions: dict[str, int] = {}
    inputs = []
    for path in paths:
        payload = load_json(path)
        manifest = payload["manifest"]
        source = manifest["source"]
        expected_questions[source] = max(
            expected_questions.get(source, 0),
            int(manifest["questions"]),
        )
        inputs.append(
            {
                "path": str(path),
                "source": source,
                "model": manifest["model"],
                "top_k": manifest["top_k"],
                "rows": len(payload["rows"]),
                "complete": manifest.get("complete"),
            }
        )
        for row in payload["rows"]:
            key = (source, int(row["question_index"]), row["strategy"])
            existing = merged.get(key)
            enriched = {**row, "source": source}
            if existing is not None and existing != enriched:
                raise ValueError(f"Conflicting duplicate generation row: {key}")
            merged[key] = enriched
    return list(merged.values()), expected_questions, inputs


def evidence_available(
    strategy: str,
    retrieval_row: dict[str, Any],
    top_k: int,
) -> bool:
    if strategy == "bm25":
        rank = retrieval_row["first_answer_rank"]
        return rank is not None and rank <= top_k
    if strategy == "long_context":
        return bool(retrieval_row["answer_present_in_full_context"])
    raise ValueError(f"Unsupported strategy: {strategy}")


def safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_group(
    source: str,
    strategy: str,
    rows: list[dict[str, Any]],
    retrieval_by_key: dict[tuple[str, int], dict[str, Any]],
    expected_questions: int,
    top_k: int,
) -> dict[str, Any]:
    outcomes = defaultdict(int)
    for row in rows:
        retrieval = retrieval_by_key[(source, int(row["question_index"]))]
        evidence = evidence_available(strategy, retrieval, top_k)
        correct = bool(row["substring_exact_match"])
        outcomes[
            f"{'evidence' if evidence else 'no_evidence'}_"
            f"{'correct' if correct else 'incorrect'}"
        ] += 1

    evidence_count = (
        outcomes["evidence_correct"] + outcomes["evidence_incorrect"]
    )
    return {
        "source": source,
        "strategy": strategy,
        "questions": len(rows),
        "expected_questions": expected_questions,
        "completion": len(rows) / expected_questions,
        "substring_exact_match": safe_mean(
            [float(row["substring_exact_match"]) for row in rows]
        ),
        "exact_match": safe_mean(
            [float(row["exact_match"]) for row in rows]
        ),
        "token_f1": safe_mean([float(row["token_f1"]) for row in rows]),
        "mean_latency_ms": safe_mean(
            [float(row["latency_ms"]) for row in rows]
        ),
        "mean_prompt_tokens": safe_mean(
            [float(row["prompt_tokens"]) for row in rows]
        ),
        "total_prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
        "total_output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "literal_evidence_rate": evidence_count / len(rows),
        "reader_success_given_literal_evidence": (
            outcomes["evidence_correct"] / evidence_count
            if evidence_count
            else 0.0
        ),
        "outcomes": dict(outcomes),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    retrieval = load_json(args.retrieval)
    retrieval_by_key = {
        (row["source"], int(row["question_index"])): row
        for row in retrieval["rows"]
    }
    generation_rows, expected_questions, inputs = merge_generation_rows(
        args.generation
    )
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in generation_rows:
        grouped[(row["source"], row["strategy"])].append(row)

    top_k_by_source = {}
    for item in inputs:
        prior = top_k_by_source.setdefault(item["source"], item["top_k"])
        if prior != item["top_k"]:
            raise ValueError(f"Inconsistent top_k for {item['source']}")

    summaries = [
        summarize_group(
            source,
            strategy,
            sorted(rows, key=lambda row: row["question_index"]),
            retrieval_by_key,
            expected_questions[source],
            top_k_by_source[source],
        )
        for (source, strategy), rows in sorted(grouped.items())
    ]
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_file": str(args.retrieval),
            "generation_inputs": inputs,
            "generation_rows": len(generation_rows),
            "sources": len({row["source"] for row in generation_rows}),
            "metric_scope": (
                "Literal answer evidence at the generation top-k, paired with "
                "the official substring exact-match generation metric. A model "
                "can answer correctly without a literal match through inference, "
                "and literal evidence does not guarantee sufficient multi-hop "
                "support."
            ),
        },
        "summaries": summaries,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
