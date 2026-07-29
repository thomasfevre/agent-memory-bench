#!/usr/bin/env python3
"""Publish compact Priority 2 records without exposing benchmark prompts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
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
MAB_SCALE = (
    RESULTS / "P2-MEMORYAGENTBENCH-CODEX-SCALE-15Q-20260729.json"
)
MEMGYM = RESULTS / "P2-MEMGYM-DR-CODEX-30X4-20260729.json"
COGNEE_PRIMARY = (
    RESULTS / "P2-COGNEE-GRAPHRAG-BENCH-SLICE-20260729.json"
)
COGNEE_SERIAL = (
    RESULTS
    / "P2-COGNEE-GRAPHRAG-BENCH-SLICE-LOCAL-SERIAL-20260729.json"
)
GRAPHITI = (
    RESULTS / "P2-GRAPHITI-GRAPHRAG-BENCH-SLICE-20260729.json"
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
            "bound, and this run has no no-context memorization control. Tool "
            "avoidance was prompt-enforced; no tool traces were found in retained "
            "stderr, but command-level disabling was added only afterward."
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


def mab_scale_record(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload["manifest"]
    series = []
    for row in payload["summaries"]:
        series.append(
            {
                "variant": row["source"].replace("factconsolidation_", ""),
                "architecture": row["architecture"],
                "attempted_calls": row["attempted_calls"],
                "provider_success_rate": rounded(
                    row["provider_success_rate"]
                ),
                "substring_accuracy": (
                    rounded(row["substring_exact_match"])
                    if "substring_exact_match" in row
                    else None
                ),
                "strict_exact_accuracy": (
                    rounded(row["exact_match"])
                    if "exact_match" in row
                    else None
                ),
                "token_f1": (
                    rounded(row["token_f1"])
                    if "token_f1" in row
                    else None
                ),
                "context_words": (
                    round(row["mean_context_words"])
                    if "mean_context_words" in row
                    else None
                ),
                "tokens_per_call": (
                    round(row["mean_tokens_used"])
                    if row.get("mean_tokens_used") is not None
                    else None
                ),
                "latency_s": (
                    rounded(row["mean_reader_latency_seconds"], 2)
                    if "mean_reader_latency_seconds" in row
                    else None
                ),
            }
        )
    return {
        "id": "memoryagentbench-conflict-scale-generation-20260729",
        "date": datetime.fromisoformat(manifest["created_at"]).date().isoformat(),
        "phase": "generation",
        "dataset": "MemoryAgentBench Conflict Resolution",
        "task": "Read contradictory memory from 32k through 262k scale",
        "method": "Full context, BM25 top-20 and hybrid top-20",
        "reader": "gpt-5.6-sol",
        "evidence_level": "official-data",
        "sample": (
            "15 fixed questions across six multi-hop and single-hop variants"
        ),
        "repetitions": manifest["repetitions"],
        "metrics": {"series": series},
        "budget": {
            "expected_calls": manifest["expected_calls"],
            "attempted_calls": manifest["attempted_unique_calls"],
            "successful_calls": manifest["successful_unique_calls"],
            "failed_calls": manifest["failed_unique_calls"],
            "top_k": manifest["top_k"],
        },
        "conclusion": (
            "Single-hop retrieval remained useful as memory grew, while "
            "multi-hop reading stayed weak; raw 262k contexts exceeded the "
            "Codex input-character ceiling but compressed retrieval paths ran."
        ),
        "limitation": (
            "Only 15 fixed questions per variant were evaluated, and the "
            "official substring metric can reward overbroad contradictory "
            "answers. Tool avoidance was prompt-enforced; no tool traces were "
            "found in retained stderr, but command-level disabling was added "
            "only afterward."
        ),
        "evidence_files": [
            "results/P2-MEMORYAGENTBENCH-CODEX-SCALE-15Q-20260729.json"
        ],
    }


def graph_engine_record(
    payload: dict[str, Any],
    *,
    run_id: str,
    evidence_file: str,
    variant: str,
) -> dict[str, Any]:
    ingestion_error = payload.get("ingestion_error")
    ingestion_errors = payload.get("ingestion_errors", [])
    documents_requested = payload["documents"]
    documents_completed = (
        None
        if ingestion_error
        else max(0, documents_requested - len(ingestion_errors))
    )
    if ingestion_error:
        operational_status = "ingestion_timeout"
    elif ingestion_errors:
        operational_status = "partial_index"
    else:
        operational_status = "complete"

    retrieval = payload.get("metrics")
    if retrieval is not None:
        retrieval = {
            key: rounded(value)
            if isinstance(value, float)
            else value
            for key, value in retrieval.items()
        }
    rows = payload.get("rows", [])
    queries_completed = sum(1 for row in rows if not row.get("error"))
    llm_tokens = payload.get("llm_tokens", {}).get("total")

    if operational_status == "ingestion_timeout":
        conclusion = (
            f"{payload['system']} did not finish ingestion inside the shared "
            "30-minute ceiling, so retrieval quality was not scored."
        )
    elif operational_status == "partial_index":
        conclusion = (
            f"{payload['system']} built {documents_completed} of "
            f"{documents_requested} documents inside the shared ingestion "
            "budget; retrieval metrics therefore describe a partial index."
        )
    else:
        conclusion = (
            f"{payload['system']} completed ingestion and retrieval inside "
            "the aligned local-engine protocol."
        )

    return {
        "id": run_id,
        "date": "2026-07-29",
        "phase": "retrieval",
        "dataset": "GraphRAG-Benchmark Novel-30752 aligned slice",
        "task": "Retrieve evidence for complex reasoning and fact questions",
        "method": (
            f"{payload['system']} {variant}, {payload['backend']}, "
            f"top-{payload['top_k']}"
        ),
        "reader": payload["model"],
        "evidence_level": "official-data",
        "sample": "20 source chunks and 10 questions",
        "repetitions": 1,
        "metrics": {
            "operational_status": operational_status,
            "documents_requested": documents_requested,
            "documents_completed": documents_completed,
            "ingestion_seconds": rounded(payload["ingestion_seconds"]),
            "ingestion_error": ingestion_error,
            "ingestion_error_count": len(ingestion_errors),
            "queries_completed": queries_completed,
            "retrieval": retrieval,
            "llm_tokens": llm_tokens,
        },
        "budget": {
            **payload["budget"],
            "top_k": payload["top_k"],
            "model": payload["model"],
            "embedding_model": payload["embedding_model"],
        },
        "conclusion": conclusion,
        "limitation": (
            "This is one local model, one 20-document slice and one repetition. "
            "A timeout is an operational-capacity result, not evidence that "
            "the engine can never achieve good retrieval quality."
        ),
        "evidence_files": [evidence_file],
    }


def memgym_record(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload["manifest"]
    series = []
    judge_scores: dict[str, list[float]] = defaultdict(list)
    for row in payload["summaries"]:
        score = row.get("mean_judge_score")
        if score is not None:
            judge_scores[row["architecture"]].append(float(score))
        series.append(
            {
                "stratum": row["stratum"],
                "architecture": row["architecture"],
                "attempted_calls": row["attempted_calls"],
                "successful_reader_calls": row["successful_reader_calls"],
                "judged_calls": row["judged_calls"],
                "exact_match": rounded(row.get("exact_match", 0.0)),
                "substring_match": rounded(
                    row.get("substring_match", 0.0)
                ),
                "token_f1": rounded(row["mean_token_f1"]),
                "provisional_judge_score": (
                    rounded(score) if score is not None else None
                ),
                "context_words": round(row["mean_context_words"]),
                "reader_latency_s": rounded(
                    row["mean_reader_latency_seconds"], 2
                ),
                "reader_tokens": round(row["mean_reader_tokens"]),
            }
        )
    macro = {
        architecture: rounded(sum(values) / len(values))
        for architecture, values in sorted(judge_scores.items())
        if values
    }
    best_architecture = (
        max(macro, key=macro.get) if macro else "unavailable"
    )
    best_score = macro.get(best_architecture)
    paired_scores: defaultdict[
        tuple[str, str, int, str], dict[str, float]
    ] = defaultdict(dict)
    for row in payload.get("rows", []):
        model_scores = [
            float(judge["response"]["score"])
            for judge in row.get("judges", [])
            if judge.get("ok")
            and isinstance(judge.get("response"), dict)
            and isinstance(judge["response"].get("score"), (int, float))
        ]
        if not model_scores:
            continue
        pair_key = (
            str(row["instance_id"]),
            str(row["stratum"]),
            int(row.get("repetition", 0)),
            str(row.get("reader_model", "")),
        )
        paired_scores[pair_key][row["architecture"]] = (
            sum(model_scores) / len(model_scores)
        )
    paired_vs_visible = {}
    for architecture in sorted(macro):
        if architecture == "visible_only":
            continue
        differences = [
            scores[architecture] - scores["visible_only"]
            for scores in paired_scores.values()
            if architecture in scores and "visible_only" in scores
        ]
        if not differences:
            continue
        paired_vs_visible[architecture] = {
            "pairs": len(differences),
            "mean_judge_difference": rounded(
                sum(differences) / len(differences)
            ),
            "wins": sum(difference > 0 for difference in differences),
            "ties": sum(difference == 0 for difference in differences),
            "losses": sum(difference < 0 for difference in differences),
        }
    conclusion = (
        f"The provisional semantic judge favored {best_architecture}"
        + (
            f" with a macro score of {best_score:.3f}."
            if best_score is not None
            else ", but no complete judged score was available."
        )
    )
    return {
        "id": "memgym-dr-reader-judge-30x4-20260729",
        "date": datetime.fromisoformat(manifest["created_at"]).date().isoformat(),
        "phase": "generation",
        "dataset": "MemGym-DR",
        "task": "Answer multi-hop deep-research questions from bounded memory",
        "method": "Visible documents versus BM25 chunk retrieval at top-1, top-2 and top-5",
        "reader": "gpt-5.6-sol",
        "evidence_level": "official-data",
        "sample": (
            f"{manifest['per_stratum']} fixed questions in each of the "
            "3-hop, 4-hop and 5-6-hop strata"
        ),
        "repetitions": manifest["repetitions"],
        "metrics": {
            "series": series,
            "architecture_macro_judge": macro,
            "paired_vs_visible": paired_vs_visible,
        },
        "budget": {
            "expected_reader_calls": manifest["expected_reader_calls"],
            "successful_reader_calls": manifest[
                "successful_reader_calls"
            ],
            "expected_judge_calls": manifest["expected_judge_calls"],
            "successful_judge_calls": manifest[
                "successful_judge_calls"
            ],
        },
        "conclusion": conclusion,
        "limitation": (
            "The semantic score is not human-calibrated. It remains "
            "provisional until two blinded annotators complete the fixed "
            "calibration pack; exact and substring match are overly strict "
            "for these long-form answers."
        ),
        "evidence_files": [
            "results/P2-MEMGYM-DR-CODEX-30X4-20260729.json"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-registry", type=Path, default=REGISTRY)
    parser.add_argument("--base-from-head", action="store_true")
    args = parser.parse_args()

    required = [
        LONGMEMEVAL,
        MAB_CONFLICT,
        MAB_SCALE,
        MEMGYM,
        COGNEE_PRIMARY,
        COGNEE_SERIAL,
        GRAPHITI,
    ]
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
    mab_scale = json.loads(MAB_SCALE.read_text(encoding="utf-8"))
    memgym = json.loads(MEMGYM.read_text(encoding="utf-8"))
    cognee_primary = json.loads(
        COGNEE_PRIMARY.read_text(encoding="utf-8")
    )
    cognee_serial = json.loads(
        COGNEE_SERIAL.read_text(encoding="utf-8")
    )
    graphiti = json.loads(GRAPHITI.read_text(encoding="utf-8"))
    if not longmemeval["manifest"].get("complete"):
        raise RuntimeError("LongMemEval campaign is incomplete")
    if not mab_scale["manifest"].get("complete"):
        raise RuntimeError("MemoryAgentBench scale campaign is incomplete")
    if not memgym["manifest"].get("complete"):
        raise RuntimeError("MemGym campaign is incomplete")

    upsert(registry, longmemeval_record(longmemeval))
    upsert(registry, mab_conflict_record(mab_conflict))
    upsert(registry, mab_scale_record(mab_scale))
    upsert(registry, memgym_record(memgym))
    upsert(
        registry,
        graph_engine_record(
            cognee_primary,
            run_id="graphrag-bench-cognee-default-20doc-20260729",
            evidence_file=(
                "results/P2-COGNEE-GRAPHRAG-BENCH-SLICE-20260729.json"
            ),
            variant="default concurrency",
        ),
    )
    upsert(
        registry,
        graph_engine_record(
            cognee_serial,
            run_id="graphrag-bench-cognee-serial-20doc-20260729",
            evidence_file=(
                "results/P2-COGNEE-GRAPHRAG-BENCH-SLICE-LOCAL-SERIAL-20260729.json"
            ),
            variant="single-item local-provider ablation",
        ),
    )
    upsert(
        registry,
        graph_engine_record(
            graphiti,
            run_id="graphrag-bench-graphiti-20doc-20260729",
            evidence_file=(
                "results/P2-GRAPHITI-GRAPHRAG-BENCH-SLICE-20260729.json"
            ),
            variant="single-coroutine ingestion",
        ),
    )
    registry["updated_at"] = "2026-07-30"
    findings = registry["findings"]
    for finding in (
        "Chunk boundaries can change final reading quality even when evidence-session recall is unchanged.",
        "Multi-hop evidence retrieval degrades with memory scale even when single-hop retrieval remains saturated.",
        "At 262k scale, compressed retrieval remained executable while raw full contexts exceeded the reader input ceiling.",
        "On the fixed MemGym-DR slice, BM25 top-2 improved the provisional paired semantic-judge score by 0.108 over visible documents alone; top-5 was not consistently better.",
        "Under the aligned 30-minute local GraphRAG budget, Graphiti indexed 13 of 20 documents while Cognee completed neither its default nor serial ingestion path.",
    ):
        if finding not in findings:
            findings.append(finding)
    limitations = registry["limitations"]
    for limitation in (
        "MemGym semantic scores remain provisional until two blinded humans complete the fixed 40-item calibration pack.",
        "The aligned GraphRAG comparison is one 20-document slice, one local model and one repetition; ingestion failures are capacity observations, not universal engine rankings.",
    ):
        if limitation not in limitations:
            limitations.append(limitation)
    REGISTRY.write_text(
        json.dumps(registry, indent=2) + "\n",
        encoding="utf-8",
    )
    print(REGISTRY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
