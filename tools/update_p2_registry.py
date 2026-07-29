#!/usr/bin/env python3
"""Publish compact Priority 2 records without exposing benchmark prompts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REGISTRY = RESULTS / "published" / "registry.json"
LONGMEMEVAL = RESULTS / "P2-LONGMEMEVAL-ROLE-ABLATION-29PAIRS-20260729.json"
MAB_CONFLICT = (
    RESULTS / "P2-MEMORYAGENTBENCH-CONFLICT-ALL-VARIANTS-20260729.json"
)


def rounded(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def upsert(registry: dict[str, Any], record: dict[str, Any]) -> None:
    for index, existing in enumerate(registry["runs"]):
        if existing["id"] == record["id"]:
            registry["runs"][index] = record
            return
    registry["runs"].append(record)


def longmemeval_record(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload["manifest"]
    summaries = {
        row["architecture"]: row for row in payload["summaries"]
    }
    comparison = payload["architecture_comparisons"][0]
    decision = payload["architecture_decision_comparisons"][0]
    series = []
    for architecture, label in (
        ("hybrid_chunks", "Turn chunks"),
        ("hybrid_user_chunks", "User-turn chunks"),
    ):
        row = summaries[architecture]
        series.append(
            {
                "context": label,
                "overall_accuracy": rounded(row["overall_accuracy"]),
                "answer_accuracy_lower_bound": rounded(row["answer_accuracy"]),
                "decision_accuracy": rounded(row["decision_accuracy"]),
                "trap_detection": rounded(row["abstention_accuracy"]),
                "false_abstention_rate": rounded(
                    row["false_abstention_rate"]
                ),
                "tokens_per_call": round(row["mean_tokens_used"]),
                "latency_s": rounded(row["mean_latency_seconds"], 2),
            }
        )
    return {
        "id": "longmemeval-role-chunking-29pairs-20260729",
        "date": datetime.fromisoformat(manifest["created_at"]).date().isoformat(),
        "phase": "generation",
        "dataset": "LongMemEval-S",
        "task": "Answer or abstain on matched answerable and near-miss pairs",
        "method": "Hybrid retrieval over turn chunks versus user-turn chunks",
        "reader": "gpt-5.6-sol",
        "evidence_level": "official-data",
        "sample": "29 matched pairs, 58 questions per architecture",
        "repetitions": manifest["repetitions"],
        "metrics": {
            "series": series,
            "overall_accuracy_difference_user_minus_turn": rounded(
                -comparison[
                    "observed_accuracy_difference_left_minus_right"
                ]
            ),
            "overall_mcnemar_p": rounded(
                comparison["mcnemar_exact_two_sided_p"], 4
            ),
            "decision_accuracy_difference_user_minus_turn": rounded(
                -decision["observed_accuracy_difference_left_minus_right"]
            ),
            "decision_mcnemar_p": rounded(
                decision["mcnemar_exact_two_sided_p"], 4
            ),
        },
        "budget": {
            "word_budget": manifest["word_budget"],
            "expected_calls": manifest["expected_calls"],
            "successful_calls": manifest["successful_unique_calls"],
        },
        "conclusion": (
            "User-turn chunking improved deterministic overall accuracy by "
            "8.6 points, but the paired decision difference was not "
            "statistically distinguishable on 29 pairs."
        ),
        "limitation": (
            "Answer matching outside the original alias set is a strict lower "
            "bound, and this run has no no-context memorization control."
        ),
        "evidence_files": [
            "results/P2-LONGMEMEVAL-ROLE-ABLATION-29PAIRS-20260729.json"
        ],
    }


def mab_conflict_record(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload["manifest"]
    series = [
        {
            "variant": row["source"].replace("factconsolidation_", ""),
            "bm25_hit_at_20": rounded(row["bm25_hit_at_k"]["20"]),
            "iterative_hit_at_20": rounded(row["greedy_hit_at_k"]["20"]),
            "bm25_latency_ms": rounded(
                row["mean_retrieval_latency_ms"], 2
            ),
            "iterative_latency_ms": rounded(
                row["mean_greedy_latency_ms"], 2
            ),
            "context_fraction_at_20": rounded(
                row["top20_compression_ratio_words"], 5
            ),
        }
        for row in payload["sources"]
    ]
    return {
        "id": "memoryagentbench-conflict-scale-retrieval-20260729",
        "date": datetime.fromisoformat(manifest["created_at"]).date().isoformat(),
        "phase": "retrieval",
        "dataset": "MemoryAgentBench Conflict Resolution",
        "task": "Retrieve answer evidence as contradictory memory grows",
        "method": "BM25 top-20 and iterative lexical expansion",
        "reader": None,
        "evidence_level": "official-data",
        "sample": "Eight 100-question variants from 6k to 262k tokens",
        "repetitions": 1,
        "metrics": {"series": series},
        "budget": {
            "top_k": 20,
            "questions": manifest["questions"],
            "max_context_scale": "262k",
        },
        "conclusion": (
            "Single-hop evidence stayed nearly saturated at every scale, while "
            "multi-hop BM25 hit@20 fell from 41% at 6k to 13% at 262k; "
            "iterative expansion recovered some evidence at much higher latency."
        ),
        "limitation": (
            "This isolates answer-string evidence coverage before generation "
            "and is not the official end-to-end benchmark score."
        ),
        "evidence_files": [
            "results/P2-MEMORYAGENTBENCH-CONFLICT-ALL-VARIANTS-20260729.json"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-registry", type=Path, default=REGISTRY)
    parser.add_argument("--base-from-head", action="store_true")
    args = parser.parse_args()

    required = [LONGMEMEVAL, MAB_CONFLICT]
    if not args.base_from_head:
        required.append(args.base_registry)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {', '.join(missing)}")

    registry = (
        json.loads(
            subprocess.run(
                ["git", "show", "HEAD:results/published/registry.json"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        if args.base_from_head
        else json.loads(args.base_registry.read_text(encoding="utf-8"))
    )
    longmemeval = json.loads(LONGMEMEVAL.read_text(encoding="utf-8"))
    mab_conflict = json.loads(MAB_CONFLICT.read_text(encoding="utf-8"))
    if not longmemeval["manifest"].get("complete"):
        raise RuntimeError("LongMemEval campaign is incomplete")

    upsert(registry, longmemeval_record(longmemeval))
    upsert(registry, mab_conflict_record(mab_conflict))
    registry["updated_at"] = "2026-07-29"
    findings = registry["findings"]
    for finding in (
        "Chunk boundaries can change final reading quality even when evidence-session recall is unchanged.",
        "Multi-hop evidence retrieval degrades with memory scale even when single-hop retrieval remains saturated.",
    ):
        if finding not in findings:
            findings.append(finding)
    REGISTRY.write_text(
        json.dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )
    print(REGISTRY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
