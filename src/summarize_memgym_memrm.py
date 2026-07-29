#!/usr/bin/env python3
"""Compare a local MemRM rerun with the checkpoint's released eval artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs must be non-empty and aligned")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def compare_predictions(
    rerun: list[dict[str, Any]], official: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(rerun) != len(official):
        raise ValueError("Prediction counts differ")
    identity_fields = ("instance_id", "step", "source", "perturbation")

    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        result = {
            tuple(row.get(field) for field in identity_fields): row
            for row in rows
        }
        if len(result) != len(rows):
            raise ValueError("Prediction identities are not unique")
        return result

    rerun_by_identity = keyed(rerun)
    official_by_identity = keyed(official)
    if set(rerun_by_identity) != set(official_by_identity):
        missing = sorted(set(official_by_identity) - set(rerun_by_identity))
        extra = sorted(set(rerun_by_identity) - set(official_by_identity))
        raise ValueError(
            f"Prediction identity sets differ: missing={missing[:3]!r} "
            f"extra={extra[:3]!r}"
        )

    rerun_probabilities: list[float] = []
    official_probabilities: list[float] = []
    agreements = 0
    for identity in sorted(rerun_by_identity):
        local = rerun_by_identity[identity]
        released = official_by_identity[identity]
        rerun_probabilities.append(float(local["prob_safe"]))
        official_probabilities.append(float(released["prob_safe"]))
        agreements += int(local["pred_label"] == released["pred_label"])
    absolute_errors = [
        abs(local - released)
        for local, released in zip(rerun_probabilities, official_probabilities)
    ]
    return {
        "n": len(rerun),
        "prediction_agreement": agreements / len(rerun),
        "prob_safe_mae": statistics.fmean(absolute_errors),
        "prob_safe_max_abs_error": max(absolute_errors),
        "prob_safe_pearson": pearson(
            rerun_probabilities, official_probabilities
        ),
        "alignment": "identity-keyed; source row order differs",
    }


def metric_projection(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "accuracy",
        "safe_f1",
        "harmful_f1",
        "auroc",
        "ece",
        "coverage",
        "n_covered",
        "accuracy_at_threshold",
    )
    return {field: metrics.get(field) for field in fields}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--memgym-repo", type=Path, required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rerun_payload = json.loads(args.rerun.read_text())
    official_payload = json.loads(args.official.read_text())
    if len(rerun_payload["reports"]) != 1:
        raise ValueError("Expected exactly one rerun report")
    rerun_report = rerun_payload["reports"][0]
    dataset = rerun_report["dataset"]
    rerun_predictions = rerun_payload["per_row_predictions"][dataset]
    official_predictions = official_payload["per_row_predictions"]

    rerun_metrics = metric_projection(rerun_report)
    official_metrics = metric_projection(official_payload["metrics"])
    deltas = {
        field: (
            rerun_metrics[field] - official_metrics[field]
            if isinstance(rerun_metrics[field], (int, float))
            and isinstance(official_metrics[field], (int, float))
            else None
        )
        for field in rerun_metrics
    }
    script_path = Path(__file__).resolve()
    payload = {
        "protocol": "memgym-memrm-reproduction-summary-v1",
        "dataset": dataset,
        "provenance": {
            "memgym_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=args.memgym_repo,
                text=True,
            ).strip(),
            "model_snapshot": args.model_snapshot,
            "dataset_snapshot": args.dataset_snapshot,
            "rerun_sha256": sha256(args.rerun),
            "official_sha256": sha256(args.official),
            "runner_sha256": sha256(script_path),
            "python": sys.version,
        },
        "rerun_environment": {
            "device": "cpu",
            "quantization": "disabled",
            "dtype_requested_by_upstream_runner": "bfloat16",
            "bootstrap_resamples": 1000,
        },
        "released_artifact_environment": {
            "shards": official_payload.get("shards"),
            "max_length": official_payload.get("max_length"),
            "truncation": official_payload.get("truncation"),
        },
        "rerun_metrics": rerun_metrics,
        "released_artifact_metrics": official_metrics,
        "rerun_minus_released": deltas,
        "prediction_comparison": compare_predictions(
            rerun_predictions, official_predictions
        ),
        "warnings_and_limits": [
            "The released artifact records eight GPU shards; the CPU rerun disables the CLI's default NF4 quantization, so hardware and numerical paths differ.",
            "The upstream setup warning about SAFE and HARMFUL tokenization concerns predict_logits; this evaluation calls predict_logits_text with the trained Y and N completions.",
            "This evaluates MemRM classification, not the end-to-end quality of a memory system.",
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "rerun_metrics": rerun_metrics,
        "released_artifact_metrics": official_metrics,
        "prediction_comparison": payload["prediction_comparison"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
