#!/usr/bin/env python3
"""Run MemoryAgentBench Conflict Resolution through isolated Codex readers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmark import Bm25, DenseIndex, MiniLm, MODEL_DIR
from execution_order import interleaved_product
from graph_benchmark_common import write_result
from locomo_hybrid_fusion import weighted_rrf
from longmemeval_codex_generation import run_codex, sha256_file
from memoryagentbench_slice import (
    exact_match,
    normalize_answer,
    split_facts,
    substring_match,
    token_f1,
)


DEFAULT_SOURCES = ["factconsolidation_mh_6k", "factconsolidation_sh_6k"]
DEFAULT_ARCHITECTURES = ["long_context", "bm25", "hybrid"]
SUPPORTED_ARCHITECTURES = tuple(DEFAULT_ARCHITECTURES)
ABSTENTION_ANSWER = "INSUFFICIENT_EVIDENCE"
READER_PROMPT_VERSION = "mab-conflict-reader-v1"
COMPATIBILITY_FIELDS = (
    "dataset_sha256",
    "top_k",
    "ranking_depth",
    "alpha_bm25",
    "reasoning_effort",
    "dry_run",
    "schema_sha256",
    "reader_prompt_version",
    "embedding_model_sha256",
    "codex_version",
    "execution_seed",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-results", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--schema",
        type=Path,
        default=root / "config" / "codex_memory_answer.schema.json",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--models", nargs="+", default=["gpt-5.6-sol"])
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=SUPPORTED_ARCHITECTURES,
        default=DEFAULT_ARCHITECTURES,
    )
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--question-indices", nargs="+", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--alpha-bm25", type=float, default=0.5)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--provider-retries", type=int, default=2)
    parser.add_argument("--execution-seed", type=int, default=20260729)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--allow-legacy-results",
        action="store_true",
        help=(
            "Allow one-time migration of result files that predate runtime "
            "compatibility metadata."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rank_facts(
    facts: list[dict[str, Any]],
    query: str,
    architecture: str,
    encoder: MiniLm | None,
    top_k: int,
    ranking_depth: int,
    alpha_bm25: float,
    *,
    bm25_index: Bm25 | None = None,
    dense_index: DenseIndex | None = None,
) -> list[dict[str, Any]]:
    if architecture == "long_context":
        return facts
    depth = min(ranking_depth, len(facts))
    bm25 = (bm25_index or Bm25(facts)).search(query, depth)
    if architecture == "bm25":
        return bm25[:top_k]
    if architecture != "hybrid":
        raise ValueError(f"Unsupported architecture: {architecture}")
    if dense_index is None:
        if encoder is None:
            raise ValueError("hybrid retrieval requires an encoder")
        dense_index = DenseIndex(facts, encoder)
    dense = dense_index.search(query, depth)
    return weighted_rrf(
        bm25,
        dense,
        alpha=alpha_bm25,
        limit=min(top_k, depth),
    )


def context_word_count(context: list[dict[str, Any]]) -> int:
    return sum(
        len(str(row["id"]).split()) + len(str(row["text"]).split())
        for row in context
    )


def render_prompt(
    question: str,
    context: list[dict[str, Any]],
) -> str:
    evidence = "\n".join(
        f"[{row['id']}] {row['text']}"
        for row in context
    )
    return f"""You are the reader in a controlled public memory benchmark.

Use only the supplied facts. Do not use tools, files, web search, or outside
knowledge. Some questions require combining multiple facts. Return the shortest
answer that fully answers the question and cite every fact needed.

If the supplied facts are insufficient, set abstain to true, set answer to
"{ABSTENTION_ANSWER}", and return an empty evidence_ids array. Otherwise set
abstain to false and cite one or more exact fact ids. Confidence must be between
0 and 1.

FACTS
{evidence}

QUESTION
{question}
"""


def score_response(
    response: dict[str, Any] | None,
    gold_answers: list[str],
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {
            "exact_match": False,
            "substring_exact_match": False,
            "token_f1": 0.0,
            "citation_ids_valid": False,
            "formally_cited_correct": False,
            "abstained": False,
            "sentinel_valid": False,
        }
    answer = str(response.get("answer", "")).strip()
    abstained = bool(response.get("abstain"))
    evidence_ids = response.get("evidence_ids", [])
    context_ids = {str(row["id"]) for row in context}
    citation_ids_valid = (
        isinstance(evidence_ids, list)
        and all(str(item) in context_ids for item in evidence_ids)
        and (not evidence_ids if abstained else bool(evidence_ids))
    )
    sentinel_valid = (
        normalize_answer(answer) == normalize_answer(ABSTENTION_ANSWER)
        if abstained
        else normalize_answer(answer) != normalize_answer(ABSTENTION_ANSWER)
    )
    exact = not abstained and exact_match(answer, gold_answers)
    substring = not abstained and substring_match(answer, gold_answers)
    return {
        "exact_match": exact,
        "substring_exact_match": substring,
        "token_f1": 0.0 if abstained else token_f1(answer, gold_answers),
        "citation_ids_valid": citation_ids_valid,
        "formally_cited_correct": (
            substring and citation_ids_valid and sentinel_valid
        ),
        "abstained": abstained,
        "sentinel_valid": sentinel_valid,
    }


def normalize_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if "citation_ids_valid" not in normalized and "citation_valid" in normalized:
        normalized["citation_ids_valid"] = normalized["citation_valid"]
    if (
        "formally_cited_correct" not in normalized
        and "grounded_correct" in normalized
    ):
        normalized["formally_cited_correct"] = normalized["grounded_correct"]
    return normalized


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        if all(key in row for key in ("source", "model", "architecture")):
            groups[
                (row["source"], row["model"], row["architecture"])
            ].append(row)
    summaries = []
    for (source, model, architecture), attempted in sorted(groups.items()):
        selected = [row for row in attempted if row.get("ok")]
        if not selected:
            summaries.append(
                {
                    "source": source,
                    "model": model,
                    "architecture": architecture,
                    "calls": 0,
                    "attempted_calls": len(attempted),
                    "successful_calls": 0,
                    "provider_success_rate": 0.0,
                }
            )
            continue
        official_matches = [
            row for row in selected if row["substring_exact_match"]
        ]
        overbroad_matches = [
            row
            for row in official_matches
            if not row["exact_match"]
        ]
        conflict_cues = [
            row
            for row in selected
            if re.search(
                r"\bconflict(?:ing|s|ed)?\b",
                str((row.get("response") or {}).get("answer", "")),
                flags=re.I,
            )
        ]
        summaries.append(
            {
                "source": source,
                "model": model,
                "architecture": architecture,
                "calls": len(selected),
                "attempted_calls": len(attempted),
                "successful_calls": len(selected),
                "provider_success_rate": len(selected) / len(attempted),
                "substring_exact_match": statistics.fmean(
                    float(row["substring_exact_match"]) for row in selected
                ),
                "exact_match": statistics.fmean(
                    float(row["exact_match"]) for row in selected
                ),
                "official_minus_strict_accuracy_gap": (
                    statistics.fmean(
                        float(row["substring_exact_match"]) for row in selected
                    )
                    - statistics.fmean(
                        float(row["exact_match"]) for row in selected
                    )
                ),
                "overbroad_share_of_official_matches": (
                    len(overbroad_matches) / len(official_matches)
                    if official_matches
                    else 0.0
                ),
                "explicit_conflict_cue_rate": len(conflict_cues) / len(selected),
                "token_f1": statistics.fmean(
                    row["token_f1"] for row in selected
                ),
                "formally_cited_accuracy": statistics.fmean(
                    float(row["formally_cited_correct"]) for row in selected
                ),
                "citation_id_validity": statistics.fmean(
                    float(row["citation_ids_valid"]) for row in selected
                ),
                "abstention_rate": statistics.fmean(
                    float(row["abstained"]) for row in selected
                ),
                "mean_context_facts": statistics.fmean(
                    row["context_facts"] for row in selected
                ),
                "mean_context_words": statistics.fmean(
                    row["context_words"] for row in selected
                ),
                "mean_reader_latency_seconds": statistics.fmean(
                    row["latency_seconds"] for row in selected
                ),
                "mean_tokens_used": (
                    statistics.fmean(
                        row["tokens_used"]
                        for row in selected
                        if row["tokens_used"] is not None
                    )
                    if any(row["tokens_used"] is not None for row in selected)
                    else None
                ),
                "total_tokens_used": sum(
                    row["tokens_used"] or 0 for row in selected
                ),
            }
        )
    return summaries


def stability_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        if row.get("ok"):
            groups[
                (row["source"], row["model"], row["architecture"])
            ].append(row)
    output = []
    for (source, model, architecture), selected in sorted(groups.items()):
        by_question: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_question[int(row["question_index"])].append(row)
        question_groups = list(by_question.values())
        repetition_counts = sorted({len(group) for group in question_groups})
        correctness_unanimous = [
            len({bool(row["substring_exact_match"]) for row in group}) == 1
            for group in question_groups
        ]
        abstention_unanimous = [
            len({bool(row["abstained"]) for row in group}) == 1
            for group in question_groups
        ]
        answer_distinct_counts = [
            len(
                {
                    normalize_answer(
                        str((row.get("response") or {}).get("answer", ""))
                    )
                    for row in group
                }
            )
            for group in question_groups
        ]
        output.append(
            {
                "source": source,
                "model": model,
                "architecture": architecture,
                "questions": len(question_groups),
                "repetitions_per_question": (
                    repetition_counts[0]
                    if len(repetition_counts) == 1
                    else repetition_counts
                ),
                "correctness_unanimity_rate": statistics.fmean(
                    float(value) for value in correctness_unanimous
                ),
                "abstention_unanimity_rate": statistics.fmean(
                    float(value) for value in abstention_unanimous
                ),
                "answer_unanimity_rate": statistics.fmean(
                    float(count == 1) for count in answer_distinct_counts
                ),
                "mean_distinct_normalized_answers": statistics.fmean(
                    answer_distinct_counts
                ),
                "success_at_least_once_rate": statistics.fmean(
                    float(any(row["substring_exact_match"] for row in group))
                    for group in question_groups
                ),
                "success_every_time_rate": statistics.fmean(
                    float(all(row["substring_exact_match"] for row in group))
                    for group in question_groups
                ),
            }
        )
    return output


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def architecture_comparisons(
    rows: list[dict[str, Any]],
    *,
    metric: str = "substring_exact_match",
    resamples: int = 20_000,
    seed: int = 61,
) -> list[dict[str, Any]]:
    by_key = {
        (
            row["source"],
            row["model"],
            row["architecture"],
            row["question_index"],
            row["repetition"],
        ): bool(row[metric])
        for row in rows
        if row.get("ok")
    }
    sources = sorted({key[0] for key in by_key})
    models = sorted({key[1] for key in by_key})
    architectures = sorted({key[2] for key in by_key})
    comparisons = []
    for source in sources:
        for model in models:
            for left_index, left in enumerate(architectures):
                for right in architectures[left_index + 1 :]:
                    units: defaultdict[
                        int, list[tuple[bool, bool]]
                    ] = defaultdict(list)
                    for key, left_value in by_key.items():
                        (
                            row_source,
                            row_model,
                            architecture,
                            question_index,
                            repetition,
                        ) = key
                        if (
                            row_source != source
                            or row_model != model
                            or architecture != left
                        ):
                            continue
                        right_key = (
                            source,
                            model,
                            right,
                            question_index,
                            repetition,
                        )
                        if right_key in by_key:
                            units[question_index].append(
                                (left_value, by_key[right_key])
                            )
                    paired = [pair for values in units.values() for pair in values]
                    left_only = sum(a and not b for a, b in paired)
                    right_only = sum(b and not a for a, b in paired)
                    question_majority_pairs = [
                        (
                            sum(bool(a) for a, _ in values) > len(values) / 2,
                            sum(bool(b) for _, b in values) > len(values) / 2,
                        )
                        for values in units.values()
                    ]
                    majority_left_only = sum(
                        a and not b for a, b in question_majority_pairs
                    )
                    majority_right_only = sum(
                        b and not a for a, b in question_majority_pairs
                    )
                    repeated = any(len(values) > 1 for values in units.values())
                    observed = (
                        statistics.fmean(float(a) for a, _ in paired)
                        - statistics.fmean(float(b) for _, b in paired)
                        if paired
                        else 0.0
                    )
                    generator = random.Random(
                        seed
                        + sum(
                            ord(character)
                            for character in f"{source}:{model}:{left}:{right}:{metric}"
                        )
                    )
                    unit_values = list(units.values())
                    bootstrap = []
                    if unit_values:
                        for _ in range(resamples):
                            sample = [
                                generator.choice(unit_values)
                                for _ in range(len(unit_values))
                            ]
                            flattened = [pair for values in sample for pair in values]
                            bootstrap.append(
                                statistics.fmean(float(a) for a, _ in flattened)
                                - statistics.fmean(float(b) for _, b in flattened)
                            )
                        bootstrap.sort()
                    comparisons.append(
                        {
                            "source": source,
                            "model": model,
                            "metric": metric,
                            "left": left,
                            "right": right,
                            "paired_questions": len(unit_values),
                            "paired_question_repetitions": len(paired),
                            "left_only_correct": left_only,
                            "right_only_correct": right_only,
                            "question_majority_left_only_correct": majority_left_only,
                            "question_majority_right_only_correct": majority_right_only,
                            "question_majority_accuracy_difference_left_minus_right": (
                                statistics.fmean(
                                    float(a) for a, _ in question_majority_pairs
                                )
                                - statistics.fmean(
                                    float(b) for _, b in question_majority_pairs
                                )
                                if question_majority_pairs
                                else 0.0
                            ),
                            "observed_accuracy_difference_left_minus_right": observed,
                            "mcnemar_unit": (
                                "question_majority" if repeated else "question"
                            ),
                            "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                                majority_left_only,
                                majority_right_only,
                            ),
                            "question_group_bootstrap_resamples": resamples,
                            "question_group_bootstrap_95_interval": (
                                [
                                    bootstrap[int(0.025 * resamples)],
                                    bootstrap[int(0.975 * resamples) - 1],
                                ]
                                if bootstrap
                                else [0.0, 0.0]
                            ),
                        }
                    )
    return comparisons


def model_comparisons(
    rows: list[dict[str, Any]],
    *,
    metric: str = "substring_exact_match",
    resamples: int = 20_000,
    seed: int = 67,
) -> list[dict[str, Any]]:
    remapped = [
        {
            **row,
            "model": row["architecture"],
            "architecture": row["model"],
        }
        for row in rows
    ]
    comparisons = architecture_comparisons(
        remapped,
        metric=metric,
        resamples=resamples,
        seed=seed,
    )
    return [
        {
            "source": row["source"],
            "architecture": row["model"],
            "metric": row["metric"],
            "left": row["left"],
            "right": row["right"],
            "paired_questions": row["paired_questions"],
            "paired_question_repetitions": row[
                "paired_question_repetitions"
            ],
            "left_only_correct": row["left_only_correct"],
            "right_only_correct": row["right_only_correct"],
            "question_majority_left_only_correct": row[
                "question_majority_left_only_correct"
            ],
            "question_majority_right_only_correct": row[
                "question_majority_right_only_correct"
            ],
            "question_majority_accuracy_difference_left_minus_right": row[
                "question_majority_accuracy_difference_left_minus_right"
            ],
            "observed_accuracy_difference_left_minus_right": row[
                "observed_accuracy_difference_left_minus_right"
            ],
            "mcnemar_unit": row["mcnemar_unit"],
            "mcnemar_exact_two_sided_p": row["mcnemar_exact_two_sided_p"],
            "question_group_bootstrap_resamples": row[
                "question_group_bootstrap_resamples"
            ],
            "question_group_bootstrap_95_interval": row[
                "question_group_bootstrap_95_interval"
            ],
        }
        for row in comparisons
    ]


def find_samples(
    parquet: Path,
    sources: list[str],
) -> dict[str, dict[str, Any]]:
    requested = set(sources)
    selected = {
        row["metadata"]["source"]: row
        for row in pq.read_table(parquet).to_pylist()
        if row.get("metadata", {}).get("source") in requested
    }
    missing = sorted(requested - set(selected))
    if missing:
        raise ValueError(f"Missing sources: {missing}")
    return selected


def selected_question_indices(
    sample: dict[str, Any],
    question_start: int,
    questions: int,
    explicit: list[int] | None,
) -> list[int]:
    indices = (
        sorted(set(explicit))
        if explicit
        else list(
            range(
                question_start,
                min(question_start + questions, len(sample["questions"])),
            )
        )
    )
    invalid = [index for index in indices if not 0 <= index < len(sample["questions"])]
    if invalid:
        raise ValueError(f"Invalid question indices: {invalid}")
    return indices


def protocol_manifest(
    *,
    dataset_sha256: str,
    sources: list[str],
    models: list[str],
    architectures: list[str],
    question_indices: list[int],
    repetitions: int,
    top_k: int,
    ranking_depth: int,
    alpha_bm25: float,
    reasoning_effort: str,
    dry_run: bool,
    schema_sha256: str,
    reader_prompt_version: str,
    embedding_model_sha256: str,
    codex_version: str,
    execution_seed: int,
) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "sources": sources,
        "models": models,
        "architectures": architectures,
        "question_indices": question_indices,
        "repetitions": repetitions,
        "top_k": top_k,
        "ranking_depth": ranking_depth,
        "alpha_bm25": alpha_bm25,
        "reasoning_effort": reasoning_effort,
        "dry_run": dry_run,
        "schema_sha256": schema_sha256,
        "reader_prompt_version": reader_prompt_version,
        "embedding_model_sha256": embedding_model_sha256,
        "codex_version": codex_version,
        "execution_seed": execution_seed,
    }


def validate_resume_manifest(
    existing: dict[str, Any],
    requested: dict[str, Any],
) -> None:
    mismatches = {
        key: {"existing": existing.get(key), "requested": value}
        for key, value in requested.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume manifest mismatch: {mismatches}")


def validate_compatible_manifest(
    existing: dict[str, Any],
    expected: dict[str, Any],
    *,
    allow_missing: bool = False,
) -> None:
    required = [key for key in COMPATIBILITY_FIELDS if key in expected]
    missing = [key for key in required if key not in existing]
    if missing and not allow_missing:
        raise ValueError(f"missing compatibility fields: {missing}")
    mismatches = {
        key: {"existing": existing.get(key), "expected": expected[key]}
        for key in required
        if key in existing and existing.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"incompatible result manifest: {mismatches}")


def load_seed_rows(
    payloads: list[dict[str, Any]],
    allowed_run_keys: set[str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload.get("rows", []):
            run_key = row.get("run_key")
            if run_key in allowed_run_keys:
                rows[run_key] = normalize_legacy_row(row)
    return rows


def inherited_legacy_execution_sources(
    path: Path,
    payload: dict[str, Any],
) -> list[str]:
    manifest = payload.get("manifest", {})
    inherited = manifest.get("legacy_execution_order_sources")
    if manifest.get("execution_order") == "mixed-legacy-and-interleaved":
        return (
            [str(item) for item in inherited]
            if isinstance(inherited, list) and inherited
            else [str(path.resolve())]
        )
    if manifest.get("execution_seed") is None:
        return [str(path.resolve())]
    return []


def build_payload(
    *,
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
    parquet: Path,
    schema: Path,
    codex_version: str,
    expected_calls: int,
    attempted_new_calls: int,
    stopped_reason: str | None,
    retrieval_build_seconds: float,
    seed_results: list[Path],
) -> dict[str, Any]:
    attempted = len({row["run_key"] for row in rows})
    successful = len({row["run_key"] for row in rows if row.get("ok")})
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": attempted == expected_calls,
            "stopped_reason": stopped_reason,
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Conflict_Resolution",
            "source_file": str(parquet),
            "schema": str(schema),
            **protocol,
            "expected_calls": expected_calls,
            "attempted_new_calls": attempted_new_calls,
            "attempted_unique_calls": attempted,
            "successful_unique_calls": successful,
            "failed_unique_calls": attempted - successful,
            "retrieval_build_seconds": retrieval_build_seconds,
            "seed_results": [str(path.resolve()) for path in seed_results],
            "official_primary_metric": "substring_exact_match",
            "scope": (
                "Static final-context retrieval and reader evaluation on public "
                "MemoryAgentBench Conflict Resolution, isolated Codex "
                "subscription readers, deterministic scoring, no API key and "
                "no private data"
            ),
            "limitations": [
                "Codex subscription agents include orchestration overhead and are not a pinned raw API endpoint.",
                (
                    f"Only {len(protocol['sources'])} selected Conflict "
                    "Resolution variants are evaluated; this is not the full "
                    "MemoryAgentBench suite."
                ),
                "This evaluates retrieval and reading over a final static context, not incremental ingestion, updating, maintenance, or forgetting.",
                "The hybrid weight and retrieval depth are fixed rather than tuned on the test questions.",
                "Formal citation validity only checks that cited ids exist in the supplied context; it does not verify entailment or provenance.",
                "Every selected question has a gold answer, so abstention_rate is a refusal rate rather than abstention accuracy.",
                "Reader latency excludes retrieval, indexing, and orchestration setup time.",
                (
                    "Configuration order includes explicitly labeled legacy rows."
                    if protocol.get("execution_order")
                    == "mixed-legacy-and-interleaved"
                    else "Configuration order is deterministically interleaved to reduce provider-drift confounding."
                ),
                "Official answer substring matching can reward an overbroad answer, so strict exact and token F1 are reported separately.",
                "Repeated Codex calls are stochastic even though retrieval is deterministic.",
            ],
            "output": str(output),
        },
        "rows": sorted(
            rows,
            key=lambda row: (
                row["source"],
                row["model"],
                row["architecture"],
                row["repetition"],
                row["question_index"],
            ),
        ),
        "summaries": summarize(rows),
        "stability": stability_summary(rows),
        "architecture_comparisons": architecture_comparisons(rows),
        "strict_exact_architecture_comparisons": architecture_comparisons(
            rows,
            metric="exact_match",
            seed=69,
        ),
        "formally_cited_architecture_comparisons": architecture_comparisons(
            rows,
            metric="formally_cited_correct",
            seed=71,
        ),
        "model_comparisons": model_comparisons(rows),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if args.top_k < 1 or args.ranking_depth < args.top_k:
        raise ValueError("ranking-depth must be at least top-k and both must be positive")
    if not 0.0 <= args.alpha_bm25 <= 1.0:
        raise ValueError("alpha-bm25 must be between zero and one")

    parquet = args.parquet.resolve()
    schema = args.schema.resolve()
    samples = find_samples(parquet, args.sources)
    first_sample = samples[args.sources[0]]
    question_indices = selected_question_indices(
        first_sample,
        args.question_start,
        args.questions,
        args.question_indices,
    )
    for source, sample in samples.items():
        if len(sample["questions"]) != len(first_sample["questions"]):
            raise ValueError(f"Question count mismatch for {source}")

    try:
        codex_version = subprocess.run(
            [args.codex_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        codex_version = "unavailable"
    embedding_model = args.model_dir / "model.onnx"
    embedding_model_sha256 = (
        sha256_file(embedding_model)
        if "hybrid" in args.architectures
        else "not_used"
    )
    protocol = protocol_manifest(
        dataset_sha256=sha256_file(parquet),
        sources=args.sources,
        models=args.models,
        architectures=args.architectures,
        question_indices=question_indices,
        repetitions=args.repetitions,
        top_k=args.top_k,
        ranking_depth=args.ranking_depth,
        alpha_bm25=args.alpha_bm25,
        reasoning_effort=args.reasoning_effort,
        dry_run=args.dry_run,
        schema_sha256=sha256_file(schema),
        reader_prompt_version=READER_PROMPT_VERSION,
        embedding_model_sha256=embedding_model_sha256,
        codex_version=codex_version,
        execution_seed=args.execution_seed,
    )

    allowed_run_keys = {
        "|".join(
            [
                source,
                str(question_index),
                architecture,
                model,
                str(repetition),
            ]
        )
        for repetition in range(args.repetitions)
        for model in args.models
        for source in args.sources
        for architecture in args.architectures
        for question_index in question_indices
    }
    seed_payloads = [
        json.loads(path.read_text())
        for path in args.seed_results
        if path.exists()
    ]
    legacy_execution_order_sources = [
        source
        for path, payload in zip(
            [path for path in args.seed_results if path.exists()],
            seed_payloads,
            strict=True,
        )
        for source in inherited_legacy_execution_sources(path, payload)
    ]
    for seed_payload in seed_payloads:
        validate_compatible_manifest(
            seed_payload.get("manifest", {}),
            protocol,
            allow_missing=args.allow_legacy_results,
        )
    existing = load_seed_rows(seed_payloads, allowed_run_keys)
    rows: list[dict[str, Any]] = list(existing.values())
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        previous_manifest = previous.get("manifest", {})
        legacy_execution_order_sources.extend(
            inherited_legacy_execution_sources(args.output, previous)
        )
        validate_compatible_manifest(
            previous_manifest,
            protocol,
            allow_missing=args.allow_legacy_results,
        )
        resume_protocol = (
            {
                key: value
                for key, value in protocol.items()
                if key in previous_manifest
            }
            if args.allow_legacy_results
            else protocol
        )
        validate_resume_manifest(previous_manifest, resume_protocol)
        existing.update(
            load_seed_rows([previous], allowed_run_keys)
        )
        rows = list(existing.values())

    protocol["execution_order"] = (
        "mixed-legacy-and-interleaved"
        if legacy_execution_order_sources
        else "interleaved"
    )
    protocol["legacy_execution_order_sources"] = sorted(
        set(legacy_execution_order_sources)
    )

    encoder = (
        MiniLm(args.model_dir)
        if "hybrid" in args.architectures
        else None
    )
    retrieval_started = time.perf_counter()
    source_state: dict[str, dict[str, Any]] = {}
    for source in args.sources:
        facts = split_facts(samples[source]["context"])
        source_state[source] = {
            "facts": facts,
            "bm25": Bm25(facts),
            "dense": DenseIndex(facts, encoder) if encoder is not None else None,
        }
    retrieval_build_seconds = time.perf_counter() - retrieval_started

    expected_calls = (
        len(args.sources)
        * len(question_indices)
        * len(args.models)
        * len(args.architectures)
        * args.repetitions
    )
    attempted_new_calls = 0
    consecutive_failures_by_model: defaultdict[str, int] = defaultdict(int)
    stopped_reason = None

    schedule = interleaved_product(
        range(args.repetitions),
        args.models,
        args.sources,
        args.architectures,
        question_indices,
        seed=args.execution_seed,
    )
    for repetition, model, source, architecture, question_index in schedule:
        # Preserve the existing stop cascade around one globally interleaved
        # task order.
        for _campaign_scope in (None,):
            for _model_scope in (None,):
                sample = samples[source]
                state = source_state[source]
                for _source_scope in (None,):
                    for _architecture_scope in (None,):
                        run_key = "|".join(
                            [
                                source,
                                str(question_index),
                                architecture,
                                model,
                                str(repetition),
                            ]
                        )
                        previous = existing.get(run_key)
                        if previous and (
                            previous.get("ok") or not args.retry_errors
                        ):
                            continue
                        if (
                            args.max_calls is not None
                            and attempted_new_calls >= args.max_calls
                        ):
                            stopped_reason = "max_calls"
                            break
                        question = sample["questions"][question_index]
                        gold_answers = sample["answers"][question_index]
                        context = rank_facts(
                            state["facts"],
                            question,
                            architecture,
                            encoder,
                            args.top_k,
                            args.ranking_depth,
                            args.alpha_bm25,
                            bm25_index=state["bm25"],
                            dense_index=state["dense"],
                        )
                        prompt = render_prompt(question, context)
                        result = (
                            {
                                "ok": False,
                                "error": "dry_run",
                                "response": None,
                                "latency_seconds": 0.0,
                                "tokens_used": 0,
                                "raw_output_sha256": None,
                                "stderr_tail": "",
                            }
                            if args.dry_run
                            else run_codex(
                                args.codex_bin,
                                model,
                                args.reasoning_effort,
                                schema,
                                prompt,
                                args.timeout_seconds,
                                args.provider_retries,
                            )
                        )
                        score = score_response(
                            result.get("response"),
                            gold_answers,
                            context,
                        )
                        row = {
                            "run_key": run_key,
                            "source": source,
                            "question_index": question_index,
                            "question": question,
                            "gold_answers": gold_answers,
                            "model": model,
                            "architecture": architecture,
                            "repetition": repetition,
                            "context_facts": len(context),
                            "context_words": context_word_count(context),
                            "context_ids": [row["id"] for row in context],
                            "literal_answer_evidence": any(
                                any(
                                    normalize_answer(answer)
                                    in normalize_answer(str(fact["text"]))
                                    for answer in gold_answers
                                )
                                for fact in context
                            ),
                            "prompt_sha256": hashlib.sha256(
                                prompt.encode()
                            ).hexdigest(),
                            **result,
                            **score,
                        }
                        existing[run_key] = row
                        rows = list(existing.values())
                        attempted_new_calls += 1
                        consecutive_failures_by_model[model] = (
                            0
                            if row["ok"]
                            else consecutive_failures_by_model[model] + 1
                        )
                        payload = build_payload(
                            protocol=protocol,
                            rows=rows,
                            output=args.output.resolve(),
                            parquet=parquet,
                            schema=schema,
                            codex_version=codex_version,
                            expected_calls=expected_calls,
                            attempted_new_calls=attempted_new_calls,
                            stopped_reason=None,
                            retrieval_build_seconds=retrieval_build_seconds,
                            seed_results=args.seed_results,
                        )
                        write_result(args.output, payload)
                        answer = (
                            row.get("response", {}).get("answer", "")
                            if isinstance(row.get("response"), dict)
                            else ""
                        )
                        print(
                            f"{source} q={question_index:02d} "
                            f"{architecture:<12} {model} r={repetition} "
                            f"ok={int(bool(row['ok']))} "
                            f"match={int(row['substring_exact_match'])} "
                            f"answer={str(answer)[:60]!r}",
                            flush=True,
                        )
                        if consecutive_failures_by_model[model] >= 3:
                            stopped_reason = (
                                f"three_consecutive_failures:{model}"
                            )
                            break
                    if stopped_reason:
                        break
                if stopped_reason:
                    break
            if stopped_reason:
                break
        if stopped_reason:
            break

    rows = list(existing.values())
    final_payload = build_payload(
        protocol=protocol,
        rows=rows,
        output=args.output.resolve(),
        parquet=parquet,
        schema=schema,
        codex_version=codex_version,
        expected_calls=expected_calls,
        attempted_new_calls=attempted_new_calls,
        stopped_reason=stopped_reason,
        retrieval_build_seconds=retrieval_build_seconds,
        seed_results=args.seed_results,
    )
    write_result(args.output, final_payload)
    print(args.output)


if __name__ == "__main__":
    main()
