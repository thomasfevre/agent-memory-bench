from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from graph_benchmark_common import write_result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def exact_mcnemar_p_value(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(
        math.comb(discordant, index)
        for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = {
        (int(row["question_index"]), row["strategy"]): row
        for row in payload["rows"]
    }
    question_indices = sorted(
        {question_index for question_index, _strategy in rows}
    )
    expected_strategies = {"bm25", "long_context"}
    missing = [
        {
            "question_index": question_index,
            "missing_strategies": sorted(
                expected_strategies
                - {
                    strategy
                    for candidate_index, strategy in rows
                    if candidate_index == question_index
                }
            ),
        }
        for question_index in question_indices
        if any(
            (question_index, strategy) not in rows
            for strategy in expected_strategies
        )
    ]
    if missing:
        raise ValueError(f"incomplete paired rows: {missing}")

    outcome_counts = {
        "both_correct": 0,
        "bm25_only": 0,
        "long_context_only": 0,
        "neither_correct": 0,
    }
    per_decile: list[dict[str, Any]] = []
    for question_index in question_indices:
        bm25_correct = bool(
            rows[(question_index, "bm25")]["substring_exact_match"]
        )
        long_correct = bool(
            rows[(question_index, "long_context")]["substring_exact_match"]
        )
        if bm25_correct and long_correct:
            outcome_counts["both_correct"] += 1
        elif bm25_correct:
            outcome_counts["bm25_only"] += 1
        elif long_correct:
            outcome_counts["long_context_only"] += 1
        else:
            outcome_counts["neither_correct"] += 1

    for start in range(0, len(question_indices), 10):
        selected = question_indices[start : start + 10]
        per_decile.append(
            {
                "range": f"{selected[0]}-{selected[-1]}",
                "questions": len(selected),
                "bm25_correct": sum(
                    bool(rows[(index, "bm25")]["substring_exact_match"])
                    for index in selected
                ),
                "long_context_correct": sum(
                    bool(
                        rows[(index, "long_context")][
                            "substring_exact_match"
                        ]
                    )
                    for index in selected
                ),
            }
        )

    strategy_rows = {
        strategy: [
            rows[(question_index, strategy)]
            for question_index in question_indices
        ]
        for strategy in expected_strategies
    }
    strategy_summary = {}
    for strategy, selected in strategy_rows.items():
        successes = sum(bool(row["substring_exact_match"]) for row in selected)
        interval = wilson_interval(successes, len(selected))
        strategy_summary[strategy] = {
            "questions": len(selected),
            "successes": successes,
            "substring_exact_match": successes / len(selected),
            "wilson_95_interval": {
                "low": interval[0],
                "high": interval[1],
            },
            "mean_latency_ms": sum(
                float(row["latency_ms"]) for row in selected
            )
            / len(selected),
            "total_prompt_tokens": sum(
                int(row["prompt_tokens"]) for row in selected
            ),
            "total_output_tokens": sum(
                int(row["output_tokens"]) for row in selected
            ),
        }

    return {
        "questions": len(question_indices),
        "paired_outcomes": outcome_counts,
        "exact_mcnemar_p_value": exact_mcnemar_p_value(
            outcome_counts["bm25_only"],
            outcome_counts["long_context_only"],
        ),
        "strategies": strategy_summary,
        "efficiency_ratios_long_context_over_bm25": {
            "mean_latency": (
                strategy_summary["long_context"]["mean_latency_ms"]
                / strategy_summary["bm25"]["mean_latency_ms"]
            ),
            "prompt_tokens": (
                strategy_summary["long_context"]["total_prompt_tokens"]
                / strategy_summary["bm25"]["total_prompt_tokens"]
            ),
        },
        "per_decile": per_decile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    write_result(args.output, analyze(payload))


if __name__ == "__main__":
    main()
