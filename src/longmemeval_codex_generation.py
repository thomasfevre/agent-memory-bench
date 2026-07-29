#!/usr/bin/env python3
"""Run a bounded LongMemEval generation campaign through Codex subscription models."""

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
import tempfile
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import Bm25, DenseIndex, MiniLm, MODEL_DIR
from graph_benchmark_common import write_result
from locomo_hybrid_fusion import weighted_rrf


DEFAULT_PAIR_IDS = [
    "f685340e",
    "6aeb4375",
    "09ba9854",
    "80ec1f4f",
    "15745da0",
    "29f2956b",
    "gpt4_70e84552",
    "gpt4_c27434e8",
]
DEFAULT_MODELS = ["gpt-5.6-luna", "gpt-5.6-sol"]
DEFAULT_ARCHITECTURES = ["no_context", "bm25_chunks", "hybrid_chunks"]
SUPPORTED_ARCHITECTURES = [
    *DEFAULT_ARCHITECTURES,
    "bm25_user_chunks",
    "hybrid_user_chunks",
]
ABSTENTION_ANSWER = "INSUFFICIENT_EVIDENCE"
TOKEN_USE_PATTERN = re.compile(r"tokens used\s+([0-9\u00a0\u202f, ]+)", re.I)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path)
    parser.add_argument("--seed-results", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--schema",
        type=Path,
        default=root / "config" / "codex_memory_answer.schema.json",
    )
    parser.add_argument(
        "--answer-aliases",
        type=Path,
        default=root / "config" / "longmemeval_codex_answer_aliases.json",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=SUPPORTED_ARCHITECTURES,
        default=DEFAULT_ARCHITECTURES,
    )
    parser.add_argument("--pair-ids", nargs="+", default=DEFAULT_PAIR_IDS)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--word-budget", type=int, default=4_000)
    parser.add_argument("--chunk-tokens", type=int, default=224)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--alpha-bm25", type=float, default=0.5)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--provider-retries", type=int, default=2)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).lower()
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9$]+", " ", text)
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def token_f1(prediction: str, reference: Any) -> float:
    predicted = normalize_text(prediction).split()
    expected = normalize_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    predicted_counts: defaultdict[str, int] = defaultdict(int)
    expected_counts: defaultdict[str, int] = defaultdict(int)
    for token in predicted:
        predicted_counts[token] += 1
    for token in expected:
        expected_counts[token] += 1
    overlap = sum(
        min(count, expected_counts[token])
        for token, count in predicted_counts.items()
    )
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def deterministic_answer_match(prediction: str, reference: Any) -> bool:
    predicted = normalize_text(prediction)
    expected = normalize_text(reference)
    if not predicted or not expected:
        return predicted == expected
    return expected in predicted or predicted in expected


def answer_matches_spec(
    prediction: str,
    reference: Any,
    spec: dict[str, Any] | None,
) -> bool:
    if not spec:
        return deterministic_answer_match(prediction, reference)
    normalized = normalize_text(prediction)
    aliases = spec.get("aliases", [])
    if aliases and any(
        normalize_text(alias) in normalized for alias in aliases
    ):
        return True
    required_all = spec.get("required_all", [])
    if required_all:
        return all(
            any(normalize_text(alias) in normalized for alias in alternatives)
            for alternatives in required_all
        )
    return False


def split_encoder_tokens(
    text: str,
    tokenizer: Any,
    chunk_tokens: int,
) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False).ids
    return [
        tokenizer.decode(token_ids[start : start + chunk_tokens]).strip()
        for start in range(0, len(token_ids), chunk_tokens)
        if token_ids[start : start + chunk_tokens]
    ]


def build_chunks(
    item: dict[str, Any],
    tokenizer: Any,
    chunk_tokens: int,
) -> list[dict[str, Any]]:
    chunks = []
    for session_id, date, session in zip(
        item["haystack_session_ids"],
        item["haystack_dates"],
        item["haystack_sessions"],
        strict=True,
    ):
        for message_index, message in enumerate(session):
            role = message.get("role", "")
            content = message.get("content", "")
            pieces = split_encoder_tokens(content, tokenizer, chunk_tokens)
            for chunk_index, text in enumerate(pieces):
                if not text:
                    continue
                chunks.append(
                    {
                        "id": f"{session_id}:m{message_index}:c{chunk_index}",
                        "session_id": session_id,
                        "date": str(date),
                        "role": role,
                        "text": text,
                    }
                )
    return chunks


def rank_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    architecture: str,
    encoder: MiniLm | None,
    ranking_depth: int,
    alpha_bm25: float,
) -> list[dict[str, Any]]:
    if architecture == "no_context":
        return []
    candidates = (
        [row for row in chunks if row["role"] == "user"]
        if "_user_" in architecture
        else chunks
    )
    depth = min(ranking_depth, len(candidates))
    bm25 = Bm25(candidates).search(query, depth)
    if architecture in {"bm25_chunks", "bm25_user_chunks"}:
        return bm25
    if architecture not in {"hybrid_chunks", "hybrid_user_chunks"} or encoder is None:
        raise ValueError(f"Unsupported architecture: {architecture}")
    dense = DenseIndex(candidates, encoder).search(query, depth)
    return weighted_rrf(
        bm25,
        dense,
        alpha=alpha_bm25,
        limit=depth,
    )


def context_header(row: dict[str, Any]) -> str:
    return f"[{row['id']}] date={row['date']} role={row['role']}"


def context_word_count(context: list[dict[str, Any]]) -> int:
    return sum(
        len(context_header(row).split()) + len(row["text"].split())
        for row in context
    )


def select_context(
    ranking: list[dict[str, Any]],
    word_budget: int,
) -> list[dict[str, Any]]:
    selected = []
    words = 0
    for row in ranking:
        header_words = len(context_header(row).split())
        remaining = word_budget - words
        if remaining <= header_words:
            continue
        content_words = row["text"].split()
        allowed_content = min(len(content_words), remaining - header_words)
        if allowed_content <= 0:
            continue
        selected.append(
            {
                **row,
                "text": " ".join(content_words[:allowed_content]),
            }
        )
        words += header_words + allowed_content
        if words >= word_budget:
            break
    return sorted(selected, key=lambda row: (row["date"], row["id"]))


def render_prompt(
    question: str,
    question_date: str,
    context: list[dict[str, Any]],
) -> str:
    context_text = "\n\n".join(
        (
            f"{context_header(row)}\n"
            f"{row['text']}"
        )
        for row in context
    )
    return f"""You are the reader in a controlled memory benchmark.

Use only the supplied context. Do not use tools, files, web search, or outside
knowledge. The context may contain a nearby fact whose subject, object, place,
time, or relationship differs from the question. Do not transfer a fact to a
different premise.

If the context directly supports the question, set abstain to false, answer
briefly, and cite one or more exact context ids in evidence_ids.

If the context does not directly support every required part of the question,
set abstain to true, set answer to "{ABSTENTION_ANSWER}", and return an empty
evidence_ids array. Confidence must be between 0 and 1.

Question date: {question_date}

CONTEXT
{context_text}

QUESTION
{question}
"""


def parse_tokens_used(stderr: str) -> int | None:
    matches = TOKEN_USE_PATTERN.findall(stderr)
    if not matches:
        return None
    digits = "".join(character for character in matches[-1] if character.isdigit())
    return int(digits) if digits else None


def run_codex(
    codex_bin: str,
    model: str,
    reasoning_effort: str,
    schema: Path,
    prompt: str,
    timeout_seconds: int,
    provider_retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="longmemeval-codex-") as directory:
        command = [
            codex_bin,
            "exec",
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "--output-schema",
            str(schema.resolve()),
            "-C",
            directory,
            "-",
        ]
        completed = None
        attempts = 0
        for attempt in range(provider_retries + 1):
            attempts = attempt + 1
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return {
                    "ok": False,
                    "error": "timeout",
                    "attempts": attempts,
                    "latency_seconds": time.perf_counter() - started,
                    "stdout": error.stdout or "",
                    "stderr_tail": (error.stderr or "")[-4_000:],
                    "tokens_used": None,
                }
            transient = (
                completed.returncode != 0
                and any(
                    marker in completed.stderr.lower()
                    for marker in (
                        "at capacity",
                        "rate limit",
                        "temporarily unavailable",
                    )
                )
            )
            if not transient or attempt >= provider_retries:
                break
            time.sleep(2 * (attempt + 1))
    assert completed is not None
    stdout = completed.stdout.strip()
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError:
        response = None
    return {
        "ok": completed.returncode == 0 and isinstance(response, dict),
        "returncode": completed.returncode,
        "attempts": attempts,
        "response": response,
        "latency_seconds": time.perf_counter() - started,
        "stderr_tail": completed.stderr[-4_000:],
        "tokens_used": parse_tokens_used(completed.stderr),
        "raw_output_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
    }


def score_response(
    response: dict[str, Any] | None,
    reference: Any,
    should_abstain: bool,
    context: list[dict[str, Any]],
    answer_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {
            "correct": False,
            "abstention_correct": False,
            "answer_match": False,
            "token_f1": 0.0,
            "citation_valid": False,
        }
    predicted_abstention = bool(response.get("abstain"))
    answer = str(response.get("answer", ""))
    evidence_ids = response.get("evidence_ids", [])
    context_ids = {row["id"] for row in context}
    sentinel_valid = (
        normalize_text(answer) == normalize_text(ABSTENTION_ANSWER)
        if predicted_abstention
        else normalize_text(answer) != normalize_text(ABSTENTION_ANSWER)
    )
    citation_valid = (
        isinstance(evidence_ids, list)
        and all(str(item) in context_ids for item in evidence_ids)
        and (not evidence_ids if predicted_abstention else bool(evidence_ids))
    )
    abstention_correct = predicted_abstention == should_abstain
    answer_match = (
        False
        if should_abstain
        else (
            not predicted_abstention
            and answer_matches_spec(answer, reference, answer_spec)
        )
    )
    return {
        "correct": (
            predicted_abstention and sentinel_valid and citation_valid
            if should_abstain
            else answer_match and citation_valid
        ),
        "abstention_correct": abstention_correct,
        "answer_match": answer_match,
        "sentinel_valid": sentinel_valid,
        "token_f1": (
            0.0 if should_abstain or predicted_abstention else token_f1(answer, reference)
        ),
        "citation_valid": citation_valid,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("ok"):
            groups[(row["model"], row["architecture"])].append(row)
    summaries = []
    for (model, architecture), selected in sorted(groups.items()):
        answerable = [row for row in selected if not row["should_abstain"]]
        abstention = [row for row in selected if row["should_abstain"]]
        summaries.append(
            {
                "model": model,
                "architecture": architecture,
                "calls": len(selected),
                "answerable_questions": len(answerable),
                "abstention_questions": len(abstention),
                "overall_accuracy": statistics.fmean(
                    float(row["correct"]) for row in selected
                ),
                "decision_accuracy": statistics.fmean(
                    float(row["abstention_correct"]) for row in selected
                ),
                "answer_accuracy": (
                    statistics.fmean(float(row["answer_match"]) for row in answerable)
                    if answerable
                    else 0.0
                ),
                "false_abstention_rate": (
                    statistics.fmean(
                        float(bool(row["response"]["abstain"]))
                        for row in answerable
                    )
                    if answerable
                    else 0.0
                ),
                "abstention_accuracy": (
                    statistics.fmean(
                        float(row["abstention_correct"]) for row in abstention
                    )
                    if abstention
                    else 0.0
                ),
                "citation_validity": statistics.fmean(
                    float(row["citation_valid"]) for row in selected
                ),
                "mean_token_f1_answerable": (
                    statistics.fmean(row["token_f1"] for row in answerable)
                    if answerable
                    else 0.0
                ),
                "mean_context_words": statistics.fmean(
                    row["context_words"] for row in selected
                ),
                "evidence_session_recall": (
                    statistics.fmean(
                        float(row["evidence_session_hit"]) for row in answerable
                    )
                    if answerable
                    else 0.0
                ),
                "mean_latency_seconds": statistics.fmean(
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


def summarize_dimensions(
    rows: list[dict[str, Any]],
    dimensions: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        if row.get("ok"):
            groups[tuple(row[dimension] for dimension in dimensions)].append(
                row
            )
    summaries = []
    for key, selected in sorted(groups.items()):
        answerable = [row for row in selected if not row["should_abstain"]]
        abstention = [row for row in selected if row["should_abstain"]]
        summaries.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "calls": len(selected),
                "overall_accuracy": statistics.fmean(
                    float(row["correct"]) for row in selected
                ),
                "decision_accuracy": statistics.fmean(
                    float(row["abstention_correct"]) for row in selected
                ),
                "answer_accuracy": (
                    statistics.fmean(float(row["answer_match"]) for row in answerable)
                    if answerable
                    else 0.0
                ),
                "abstention_accuracy": (
                    statistics.fmean(
                        float(row["abstention_correct"]) for row in abstention
                    )
                    if abstention
                    else 0.0
                ),
                "mean_latency_seconds": statistics.fmean(
                    row["latency_seconds"] for row in selected
                ),
                "total_tokens_used": sum(
                    row["tokens_used"] or 0 for row in selected
                ),
            }
        )
    return summaries


def pair_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if not row.get("ok"):
            continue
        key = (row["model"], row["architecture"])
        pair_key = f"{row['pair_id']}:{row['repetition']}"
        groups[key][pair_key].append(bool(row["correct"]))
    results = []
    for (model, architecture), pairs in sorted(groups.items()):
        complete = [values for values in pairs.values() if len(values) == 2]
        results.append(
            {
                "model": model,
                "architecture": architecture,
                "complete_pair_repetitions": len(complete),
                "both_correct": sum(all(values) for values in complete),
                "both_correct_rate": (
                    statistics.fmean(float(all(values)) for values in complete)
                    if complete
                    else 0.0
                ),
            }
        )
    return results


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
    resamples: int = 20_000,
    seed: int = 43,
    metric: str = "correct",
) -> list[dict[str, Any]]:
    by_key = {
        (
            row["model"],
            row["architecture"],
            row["question_id"],
            row["repetition"],
        ): bool(row[metric])
        for row in rows
        if row.get("ok")
    }
    models = sorted({key[0] for key in by_key})
    architectures = sorted({key[1] for key in by_key})
    comparisons = []
    for model in models:
        for left_index, left in enumerate(architectures):
            for right in architectures[left_index + 1 :]:
                units: defaultdict[tuple[str, int], list[tuple[bool, bool]]] = (
                    defaultdict(list)
                )
                for key, left_value in by_key.items():
                    row_model, architecture, question_id, repetition = key
                    if row_model != model or architecture != left:
                        continue
                    right_key = (model, right, question_id, repetition)
                    if right_key not in by_key:
                        continue
                    pair_id = question_id.removesuffix("_abs")
                    units[(pair_id, repetition)].append(
                        (left_value, by_key[right_key])
                    )
                complete_units = [
                    values for values in units.values() if len(values) == 2
                ]
                pairs = [pair for values in complete_units for pair in values]
                left_only = sum(a and not b for a, b in pairs)
                right_only = sum(b and not a for a, b in pairs)
                observed = (
                    statistics.fmean(float(a) for a, _ in pairs)
                    - statistics.fmean(float(b) for _, b in pairs)
                    if pairs
                    else 0.0
                )
                generator = random.Random(
                    seed
                    + sum(ord(character) for character in f"{model}:{left}:{right}")
                )
                bootstrap = []
                if complete_units:
                    for _ in range(resamples):
                        sampled = [
                            generator.choice(complete_units)
                            for _ in range(len(complete_units))
                        ]
                        flattened = [pair for values in sampled for pair in values]
                        bootstrap.append(
                            statistics.fmean(float(a) for a, _ in flattened)
                            - statistics.fmean(float(b) for _, b in flattened)
                        )
                    bootstrap.sort()
                comparisons.append(
                    {
                        "model": model,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "question_repetitions": len(pairs),
                        "pair_repetitions": len(complete_units),
                        "left_only_correct": left_only,
                        "right_only_correct": right_only,
                        "observed_accuracy_difference_left_minus_right": observed,
                        "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                            left_only,
                            right_only,
                        ),
                        "pair_group_bootstrap_resamples": resamples,
                        "pair_group_bootstrap_95_interval": (
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
    resamples: int = 20_000,
    seed: int = 47,
) -> list[dict[str, Any]]:
    by_key = {
        (
            row["architecture"],
            row["model"],
            row["question_id"],
            row["repetition"],
        ): bool(row["correct"])
        for row in rows
        if row.get("ok")
    }
    architectures = sorted({key[0] for key in by_key})
    models = sorted({key[1] for key in by_key})
    comparisons = []
    for architecture in architectures:
        for left_index, left in enumerate(models):
            for right in models[left_index + 1 :]:
                units: defaultdict[tuple[str, int], list[tuple[bool, bool]]] = (
                    defaultdict(list)
                )
                for key, left_value in by_key.items():
                    row_architecture, model, question_id, repetition = key
                    if row_architecture != architecture or model != left:
                        continue
                    right_key = (
                        architecture,
                        right,
                        question_id,
                        repetition,
                    )
                    if right_key not in by_key:
                        continue
                    pair_id = question_id.removesuffix("_abs")
                    units[(pair_id, repetition)].append(
                        (left_value, by_key[right_key])
                    )
                complete_units = [
                    values for values in units.values() if len(values) == 2
                ]
                pairs = [pair for values in complete_units for pair in values]
                left_only = sum(a and not b for a, b in pairs)
                right_only = sum(b and not a for a, b in pairs)
                observed = (
                    statistics.fmean(float(a) for a, _ in pairs)
                    - statistics.fmean(float(b) for _, b in pairs)
                    if pairs
                    else 0.0
                )
                generator = random.Random(
                    seed
                    + sum(
                        ord(character)
                        for character in f"{architecture}:{left}:{right}"
                    )
                )
                bootstrap = []
                if complete_units:
                    for _ in range(resamples):
                        sampled = [
                            generator.choice(complete_units)
                            for _ in range(len(complete_units))
                        ]
                        flattened = [pair for values in sampled for pair in values]
                        bootstrap.append(
                            statistics.fmean(float(a) for a, _ in flattened)
                            - statistics.fmean(float(b) for _, b in flattened)
                        )
                    bootstrap.sort()
                comparisons.append(
                    {
                        "architecture": architecture,
                        "left": left,
                        "right": right,
                        "question_repetitions": len(pairs),
                        "pair_repetitions": len(complete_units),
                        "left_only_correct": left_only,
                        "right_only_correct": right_only,
                        "observed_accuracy_difference_left_minus_right": observed,
                        "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                            left_only,
                            right_only,
                        ),
                        "pair_group_bootstrap_resamples": resamples,
                        "pair_group_bootstrap_95_interval": (
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


def load_items(dataset: Path, pair_ids: list[str]) -> list[dict[str, Any]]:
    data = json.loads(dataset.read_text())
    by_id = {item["question_id"]: item for item in data}
    selected = []
    for pair_id in pair_ids:
        for question_id in (pair_id, f"{pair_id}_abs"):
            if question_id not in by_id:
                raise KeyError(f"Missing paired question: {question_id}")
            selected.append(by_id[question_id])
    return selected


def context_cache_fingerprint(
    dataset_sha256: str,
    pair_ids: list[str],
    architectures: list[str],
    word_budget: int,
    chunk_tokens: int,
    ranking_depth: int,
    alpha_bm25: float,
) -> str:
    payload = {
        "dataset_sha256": dataset_sha256,
        "pair_ids": pair_ids,
        "architectures": architectures,
        "word_budget": word_budget,
        "chunk_tokens": chunk_tokens,
        "ranking_depth": ranking_depth,
        "alpha_bm25": alpha_bm25,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0.0 <= args.alpha_bm25 <= 1.0:
        raise ValueError("alpha-bm25 must be between zero and one")
    schema = args.schema.resolve()
    answer_aliases_path = args.answer_aliases.resolve()
    answer_aliases = json.loads(answer_aliases_path.read_text())
    dataset = args.dataset.resolve()
    items = load_items(dataset, args.pair_ids)
    dataset_sha256 = sha256_file(dataset)
    cache_fingerprint = context_cache_fingerprint(
        dataset_sha256,
        args.pair_ids,
        args.architectures,
        args.word_budget,
        args.chunk_tokens,
        args.ranking_depth,
        args.alpha_bm25,
    )
    context_cache_path = (
        args.context_cache.resolve()
        if args.context_cache
        else args.output.with_suffix(".contexts.json").resolve()
    )
    contexts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    context_cache_hit = False
    context_build_started = time.perf_counter()
    if context_cache_path.exists():
        try:
            cached = json.loads(context_cache_path.read_text())
            if cached.get("manifest", {}).get("fingerprint") == cache_fingerprint:
                contexts = {
                    (row["question_id"], row["architecture"]): row["context"]
                    for row in cached["contexts"]
                }
                context_cache_hit = True
        except (json.JSONDecodeError, KeyError, TypeError):
            contexts = {}
    if not context_cache_hit:
        encoder = MiniLm(args.model_dir)
        for item in items:
            chunks = build_chunks(
                item,
                encoder.tokenizer,
                args.chunk_tokens,
            )
            for architecture in args.architectures:
                ranking = rank_chunks(
                    chunks,
                    item["question"],
                    architecture,
                    encoder,
                    args.ranking_depth,
                    args.alpha_bm25,
                )
                contexts[(item["question_id"], architecture)] = select_context(
                    ranking,
                    args.word_budget,
                )
        write_result(
            context_cache_path,
            {
                "manifest": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "fingerprint": cache_fingerprint,
                    "dataset_sha256": dataset_sha256,
                    "scope": "Public benchmark contexts only",
                },
                "contexts": [
                    {
                        "question_id": question_id,
                        "architecture": architecture,
                        "context": context,
                    }
                    for (question_id, architecture), context in sorted(
                        contexts.items()
                    )
                ],
            },
        )
    context_build_seconds = time.perf_counter() - context_build_started

    allowed_run_keys = {
        "|".join([item["question_id"], architecture, model, str(repetition)])
        for repetition in range(args.repetitions)
        for model in args.models
        for architecture in args.architectures
        for item in items
    }
    existing: dict[str, Any] = {}
    result_sources = [
        *(path.resolve() for path in args.seed_results),
        args.output.resolve(),
    ]
    for result_source in result_sources:
        if not result_source.exists():
            continue
        try:
            previous = json.loads(result_source.read_text())
            for row in previous.get("rows", []):
                if row.get("run_key") in allowed_run_keys:
                    existing[row["run_key"]] = row
        except (json.JSONDecodeError, KeyError):
            continue

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

    rows = []
    pending_calls = 0
    consecutive_failures = 0
    stopped_reason = None
    for repetition in range(args.repetitions):
        for model in args.models:
            for architecture in args.architectures:
                for item in items:
                    run_key = "|".join(
                        [
                            item["question_id"],
                            architecture,
                            model,
                            str(repetition),
                        ]
                    )
                    previous = existing.get(run_key)
                    if previous and (previous.get("ok") or not args.retry_errors):
                        rows.append(previous)
                        continue
                    if args.max_calls is not None and pending_calls >= args.max_calls:
                        stopped_reason = "max_calls"
                        break
                    context = contexts[(item["question_id"], architecture)]
                    prompt = render_prompt(
                        item["question"],
                        str(item["question_date"]),
                        context,
                    )
                    if args.dry_run:
                        result = {
                            "ok": True,
                            "response": None,
                            "latency_seconds": 0.0,
                            "tokens_used": 0,
                            "raw_output_sha256": None,
                            "stderr_tail": "",
                        }
                    else:
                        result = run_codex(
                            args.codex_bin,
                            model,
                            args.reasoning_effort,
                            schema,
                            prompt,
                            args.timeout_seconds,
                            args.provider_retries,
                        )
                    selected_sessions = {row["session_id"] for row in context}
                    score = score_response(
                        result.get("response"),
                        item.get("answer"),
                        item["question_id"].endswith("_abs"),
                        context,
                        answer_aliases.get(item["question_id"]),
                    )
                    row = {
                        "run_key": run_key,
                        "pair_id": item["question_id"].removesuffix("_abs"),
                        "question_id": item["question_id"],
                        "question_type": item["question_type"],
                        "should_abstain": item["question_id"].endswith("_abs"),
                        "model": model,
                        "architecture": architecture,
                        "repetition": repetition,
                        "context_words": context_word_count(context),
                        "context_chunks": len(context),
                        "context_ids": [chunk["id"] for chunk in context],
                        "evidence_session_hit": bool(
                            selected_sessions & set(item["answer_session_ids"])
                        ),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        **result,
                        **score,
                    }
                    rows.append(row)
                    existing[run_key] = row
                    pending_calls += 1
                    consecutive_failures = (
                        0 if row["ok"] else consecutive_failures + 1
                    )
                    payload = {
                        "manifest": {
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "complete": False,
                            "stopped_reason": None,
                            "dataset": str(dataset),
                            "dataset_sha256": dataset_sha256,
                            "schema": str(schema),
                            "schema_sha256": sha256_file(schema),
                            "answer_aliases": str(answer_aliases_path),
                            "answer_aliases_sha256": sha256_file(
                                answer_aliases_path
                            ),
                            "codex_version": codex_version,
                            "models": args.models,
                            "architectures": args.architectures,
                            "seed_results": [
                                str(path.resolve()) for path in args.seed_results
                            ],
                            "pair_ids": args.pair_ids,
                            "questions": len(items),
                        "repetitions": args.repetitions,
                        "word_budget": args.word_budget,
                        "chunk_tokens": args.chunk_tokens,
                        "ranking_depth": args.ranking_depth,
                            "alpha_bm25": args.alpha_bm25,
                            "reasoning_effort": args.reasoning_effort,
                            "context_build_seconds": context_build_seconds,
                            "context_cache": str(context_cache_path),
                            "context_cache_hit": context_cache_hit,
                            "provider_retries": args.provider_retries,
                            "scope": (
                                "Public LongMemEval-S paired near-misses, Codex "
                                "subscription execution, deterministic scoring, "
                                "no API key and no private data"
                            ),
                        },
                        "rows": rows,
                        "summaries": summarize(rows),
                        "summaries_by_repetition": summarize_dimensions(
                            rows,
                            ("repetition", "model", "architecture"),
                        ),
                        "summaries_by_question_type": summarize_dimensions(
                            rows,
                            ("model", "architecture", "question_type"),
                        ),
                        "pair_consistency": pair_consistency(rows),
                    }
                    write_result(args.output, payload)
                    if consecutive_failures >= 3:
                        stopped_reason = "three_consecutive_failures"
                        break
                if stopped_reason:
                    break
            if stopped_reason:
                break
        if stopped_reason:
            break

    expected = (
        len(items)
        * len(args.models)
        * len(args.architectures)
        * args.repetitions
    )
    final_payload = {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": len({row["run_key"] for row in rows if row.get("ok")})
            == expected,
            "stopped_reason": stopped_reason,
            "dataset": str(dataset),
            "dataset_sha256": dataset_sha256,
            "schema": str(schema),
            "schema_sha256": sha256_file(schema),
            "answer_aliases": str(answer_aliases_path),
            "answer_aliases_sha256": sha256_file(answer_aliases_path),
            "codex_version": codex_version,
            "models": args.models,
            "architectures": args.architectures,
            "seed_results": [
                str(path.resolve()) for path in args.seed_results
            ],
            "pair_ids": args.pair_ids,
            "sample_rule": (
                "Two SHA-256 ordered pairs per represented question type "
                "with salt lme-codex-pilot-v1"
            ),
            "questions": len(items),
            "repetitions": args.repetitions,
            "expected_calls": expected,
            "attempted_new_calls": pending_calls,
            "successful_unique_calls": len(
                {row["run_key"] for row in rows if row.get("ok")}
            ),
            "word_budget": args.word_budget,
            "chunk_tokens": args.chunk_tokens,
            "ranking_depth": args.ranking_depth,
            "alpha_bm25": args.alpha_bm25,
            "reasoning_effort": args.reasoning_effort,
            "context_build_seconds": context_build_seconds,
            "context_cache": str(context_cache_path),
            "context_cache_fingerprint": cache_fingerprint,
            "context_cache_hit": context_cache_hit,
            "provider_retries": args.provider_retries,
            "scope": (
                "Public LongMemEval-S paired near-misses, Codex subscription "
                "execution, deterministic scoring, no API key and no private data"
            ),
            "limitations": [
                "Codex subscription agents include orchestration overhead and are not a pinned raw API endpoint.",
                "The eight selected pairs are a stratified diagnostic slice, not the full LongMemEval benchmark.",
                "Deterministic answer matching is stricter and less semantic than the official LLM judge.",
                "The fixed 50/50 hybrid weight is preregistered rather than tuned on this slice.",
                "LongMemEval is public, so the no-context control is required to expose possible benchmark memorization.",
            ],
        },
        "rows": rows,
        "summaries": summarize(rows),
        "summaries_by_repetition": summarize_dimensions(
            rows,
            ("repetition", "model", "architecture"),
        ),
        "summaries_by_question_type": summarize_dimensions(
            rows,
            ("model", "architecture", "question_type"),
        ),
        "pair_consistency": pair_consistency(rows),
        "architecture_comparisons": architecture_comparisons(rows),
        "architecture_decision_comparisons": architecture_comparisons(
            rows,
            seed=53,
            metric="abstention_correct",
        ),
        "model_comparisons": model_comparisons(rows),
    }
    write_result(args.output, final_payload)
    print(args.output)


if __name__ == "__main__":
    main()
