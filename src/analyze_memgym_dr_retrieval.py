#!/usr/bin/env python3
"""Paired analysis for the provider-free MemGym-DR retrieval slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    index = round((len(values) - 1) * probability)
    return sorted(values)[index]


def paired_bootstrap(
    differences: list[float], resamples: int, seed: int
) -> dict[str, float | int]:
    if not differences:
        return {
            "n": 0,
            "mean_difference": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    generator = random.Random(seed)
    means = [
        statistics.fmean(
            generator.choice(differences) for _ in differences
        )
        for _ in range(resamples)
    ]
    return {
        "n": len(differences),
        "mean_difference": statistics.fmean(differences),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def compare(
    rows: list[dict[str, Any]], top_k: int, resamples: int, seed: int
) -> dict[str, Any]:
    selected = [row for row in rows if row["top_k"] == top_k]
    by_key = {
        (row["instance_id"], row["strategy"]): row
        for row in selected
    }
    instance_ids = sorted({row["instance_id"] for row in selected})
    differences = [
        by_key[(instance_id, "ir_naive_rag")]["fact_recall"]
        - by_key[(instance_id, "ir_bm25")]["fact_recall"]
        for instance_id in instance_ids
    ]
    result = {
        "dense_minus_bm25": paired_bootstrap(
            differences, resamples, seed + top_k
        ),
        "wins_dense": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "wins_bm25": sum(value < 0 for value in differences),
    }
    for stratum in ("3hop", "4hop", "56hop"):
        stratum_ids = [
            instance_id
            for instance_id in instance_ids
            if by_key[(instance_id, "ir_bm25")]["stratum"] == stratum
        ]
        stratum_differences = [
            by_key[(instance_id, "ir_naive_rag")]["fact_recall"]
            - by_key[(instance_id, "ir_bm25")]["fact_recall"]
            for instance_id in stratum_ids
        ]
        result[stratum] = paired_bootstrap(
            stratum_differences, resamples, seed + top_k + len(stratum)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")

    source = json.loads(args.input.read_text())
    rows = source["sample_results"]
    script_path = Path(__file__).resolve()
    payload = {
        "protocol": "memgym-dr-provider-free-retrieval-paired-analysis-v1",
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "resamples": args.resamples,
        "seed": args.seed,
        "comparisons": {
            str(top_k): compare(rows, top_k, args.resamples, args.seed)
            for top_k in source["configuration"]["top_k"]
        },
        "interpretation_guardrail": (
            "The paired bootstrap applies to the seeded 90-instance sample "
            "and the repository's lexical fact-recall proxy only."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["comparisons"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
