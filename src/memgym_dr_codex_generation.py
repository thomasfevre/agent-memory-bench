#!/usr/bin/env python3
"""Run a bounded MemGym-DR reader and judge through Codex subscriptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from execution_order import interleaved_product
from graph_benchmark_common import write_result
from longmemeval_codex_generation import run_codex, sha256_file
from memoryagentbench_slice import (
    exact_match,
    substring_match,
    token_f1,
)


STRATA = {
    "3hop": "3hop_verified.jsonl",
    "4hop": "4hop_paper_run.jsonl",
    "56hop": "56hop_clean.jsonl",
}
SUPPORTED_ARCHITECTURES = (
    "visible_only",
    "bm25_k1",
    "bm25_k2",
    "bm25_k5",
)
READER_PROMPT_VERSION = "memgym-dr-reader-v1"
JUDGE_PROMPT_VERSION = "memgym-dr-official-judge-v1"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--answer-schema",
        type=Path,
        default=root / "config" / "codex_memgym_answer.schema.json",
    )
    parser.add_argument(
        "--judge-schema",
        type=Path,
        default=root / "config" / "codex_memgym_judge.schema.json",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--reader-models", nargs="+", default=["gpt-5.6-sol"])
    parser.add_argument("--judge-models", nargs="+", default=["gpt-5.6-luna"])
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=SUPPORTED_ARCHITECTURES,
        default=list(SUPPORTED_ARCHITECTURES),
    )
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--provider-retries", type=int, default=2)
    parser.add_argument("--execution-seed", type=int, default=20260729)
    parser.add_argument("--max-reader-calls", type=int)
    parser.add_argument("--max-judge-calls", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def selected_ids(
    sample_manifest: dict[str, Any],
    per_stratum: int,
) -> dict[str, list[str]]:
    available = sample_manifest["configuration"]["sample_ids"]
    selected = {
        stratum: sorted(
            available[stratum],
            key=lambda instance_id: hashlib.sha256(
                (
                    f"{READER_PROMPT_VERSION}|{stratum}|{instance_id}"
                ).encode()
            ).hexdigest(),
        )[:per_stratum]
        for stratum in STRATA
    }
    if any(len(ids) != per_stratum for ids in selected.values()):
        raise ValueError("sample manifest does not contain enough ids")
    return selected


def load_selected_rows(
    dataset_dir: Path,
    ids_by_stratum: dict[str, list[str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for stratum, filename in STRATA.items():
        wanted = set(ids_by_stratum[stratum])
        with (dataset_dir / filename).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                instance_id = row.get("instance_id")
                if instance_id in wanted:
                    selected[(stratum, instance_id)] = row
                    if sum(key[0] == stratum for key in selected) == len(wanted):
                        break
        missing = wanted - {
            instance_id
            for candidate_stratum, instance_id in selected
            if candidate_stratum == stratum
        }
        if missing:
            raise ValueError(
                f"missing {stratum} instances: {', '.join(sorted(missing))}"
            )
    return selected


def visible_documents(instance: dict[str, Any], turn_index: int) -> list[dict[str, Any]]:
    policy = instance.get("eviction_policy") or {}
    if policy.get("mode", "full_eviction") == "full_eviction":
        visible_start = turn_index
    else:
        window = int(policy.get("window_size", 1))
        visible_start = max(0, turn_index - window + 1)
    return [
        document
        for index in range(visible_start, turn_index + 1)
        for document in instance["turns"][index].get("documents", [])
    ]


def format_documents(documents: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{index}] {document.get('title', 'Untitled')}\n"
        f"{document.get('text', '')}"
        for index, document in enumerate(documents, start=1)
    )


def build_observations(instance: dict[str, Any]) -> list[str]:
    observations = []
    for turn_index, turn in enumerate(instance["turns"]):
        parts = [
            f"=== Turn {turn_index + 1}: {turn.get('sub_query', '')} ==="
        ]
        for document in visible_documents(instance, turn_index):
            parts.append(f"--- {document.get('title', 'Untitled')} ---")
            parts.append(str(document.get("text", "")))
        observations.append("\n\n".join(parts))
    return observations


def chunk_observations(
    observations: list[str],
    chunk_size_tokens: int = 512,
) -> list[str]:
    words_per_chunk = max(1, int(chunk_size_tokens * 0.75))
    return [
        " ".join(words[start : start + words_per_chunk])
        for observation in observations
        for words in [observation.split()]
        for start in range(0, len(words), words_per_chunk)
        if words[start : start + words_per_chunk]
    ]


def retrieve_notes(instance: dict[str, Any], architecture: str) -> str:
    if architecture == "visible_only":
        return ""
    top_k = int(architecture.removeprefix("bm25_k"))
    chunks = chunk_observations(build_observations(instance))
    if not chunks:
        return "(no relevant chunks found)"
    index = BM25Okapi([chunk.lower().split() for chunk in chunks])
    scores = index.get_scores(instance["question"].lower().split())
    positions = sorted(
        range(len(scores)),
        key=lambda position: scores[position],
        reverse=True,
    )[:top_k]
    selected = [chunks[position] for position in positions if scores[position] > 0]
    return (
        "\n\n---\n\n".join(selected)
        if selected
        else "(no relevant chunks found)"
    )


def render_reader_prompt(instance: dict[str, Any], notes: str) -> str:
    visible = visible_documents(instance, len(instance["turns"]) - 1)
    visible_text = format_documents(visible)
    visible_section = (
        f"Documents still visible from recent turns:\n{visible_text}"
        if visible_text.strip()
        else "No documents are currently visible."
    )
    return f"""You are an AI research assistant. You have been conducting a multi-step search to answer a question. Earlier search results have been removed from your context, but you took notes along the way.

Question: {instance["question"]}

Your accumulated notes from previous turns:
{notes if notes else "(no notes taken)"}

{visible_section}

Using your notes and any visible documents above, answer the question.
If your notes don't contain enough information, say "insufficient information".

Provide only the structured response requested by the output schema.
"""


def render_judge_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: str,
) -> str:
    return f"""You are an answer judge. Compare the predicted answer to the gold answer.
Score 1.0 if the predicted answer is correct (semantically equivalent), 0.0 if wrong.
Partial credit (0.3-0.7) for partially correct answers.

Gold answer: {gold_answer}
Predicted answer: {predicted_answer}
Question: {question}

Provide only the structured response requested by the output schema.
"""


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        groups[(row["stratum"], row["reader_model"], row["architecture"])].append(
            row
        )
    summaries = []
    for (stratum, reader, architecture), attempted in sorted(groups.items()):
        successful = [row for row in attempted if row.get("reader_ok")]
        judge_scores = [
            float(judge["response"]["score"])
            for row in successful
            for judge in row.get("judges", [])
            if judge.get("ok") and isinstance(judge.get("response"), dict)
        ]
        summaries.append(
            {
                "stratum": stratum,
                "reader": reader,
                "architecture": architecture,
                "attempted_calls": len(attempted),
                "successful_reader_calls": len(successful),
                "provider_success_rate": (
                    len(successful) / len(attempted) if attempted else 0.0
                ),
                "exact_match": (
                    statistics.fmean(float(row["exact_match"]) for row in successful)
                    if successful
                    else 0.0
                ),
                "substring_match": (
                    statistics.fmean(
                        float(row["substring_match"]) for row in successful
                    )
                    if successful
                    else 0.0
                ),
                "mean_token_f1": (
                    statistics.fmean(row["token_f1"] for row in successful)
                    if successful
                    else 0.0
                ),
                "mean_judge_score": (
                    statistics.fmean(judge_scores) if judge_scores else None
                ),
                "judged_calls": len(judge_scores),
                "mean_context_words": (
                    statistics.fmean(row["context_words"] for row in successful)
                    if successful
                    else 0.0
                ),
                "mean_reader_latency_seconds": (
                    statistics.fmean(
                        row["reader_latency_seconds"] for row in successful
                    )
                    if successful
                    else 0.0
                ),
                "mean_reader_tokens": (
                    statistics.fmean(
                        row["reader_tokens"] or 0 for row in successful
                    )
                    if successful
                    else 0.0
                ),
            }
        )
    return summaries


def save_payload(
    output: Path,
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_reader_calls: int,
) -> None:
    unique_reader_successes = sum(bool(row.get("reader_ok")) for row in rows)
    expected_judge_calls = (
        unique_reader_successes * len(protocol["judge_models"])
    )
    judge_attempts = sum(
        len(
            {
                judge["model"]
                for judge in row.get("judges", [])
                if judge.get("model") in protocol["judge_models"]
            }
        )
        for row in rows
        if row.get("reader_ok")
    )
    judge_successes = sum(
        bool(judge.get("ok"))
        for row in rows
        if row.get("reader_ok")
        for judge in row.get("judges", [])
        if judge.get("model") in protocol["judge_models"]
    )
    reader_schedule_complete = len(rows) == expected_reader_calls
    judge_schedule_complete = judge_attempts == expected_judge_calls
    write_result(
        output,
        {
            "manifest": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                **protocol,
                "expected_reader_calls": expected_reader_calls,
                "reader_rows": len(rows),
                "successful_reader_calls": unique_reader_successes,
                "failed_reader_calls": len(rows) - unique_reader_successes,
                "expected_judge_calls": expected_judge_calls,
                "attempted_judge_calls": judge_attempts,
                "successful_judge_calls": judge_successes,
                "failed_judge_calls": judge_attempts - judge_successes,
                "reader_schedule_complete": reader_schedule_complete,
                "judge_schedule_complete": judge_schedule_complete,
                "complete": (
                    reader_schedule_complete and judge_schedule_complete
                ),
                "scope": (
                    "Public MemGym-DR instances, official BM25 chunking and "
                    "judge prompt, isolated Codex subscription readers"
                ),
                "limitations": [
                    "The reader and judge are subscription agents, not pinned raw API endpoints.",
                    "This bounded sample is a SHA-256 subsample of an already seeded stratified retrieval sample.",
                    "LLM judge calibration against blinded human labels is reported separately.",
                ],
            },
            "rows": rows,
            "summaries": summarize(rows),
        },
    )


def main() -> int:
    args = parse_args()
    if args.per_stratum < 1 or args.repetitions < 1:
        raise ValueError("per-stratum and repetitions must be positive")
    dataset_dir = args.dataset_dir.resolve()
    sample_manifest_path = args.sample_manifest.resolve()
    output = args.output.resolve()
    answer_schema = args.answer_schema.resolve()
    judge_schema = args.judge_schema.resolve()
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    ids_by_stratum = selected_ids(sample_manifest, args.per_stratum)
    instances = load_selected_rows(dataset_dir, ids_by_stratum)
    codex_version = subprocess.run(
        [args.codex_bin, "--version"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    ).stdout.strip()
    protocol = {
        "protocol": "memgym-dr-codex-reader-judge-v1",
        "dataset_sha256": {
            stratum: sha256_file(dataset_dir / filename)
            for stratum, filename in STRATA.items()
        },
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "answer_schema_sha256": sha256_file(answer_schema),
        "judge_schema_sha256": sha256_file(judge_schema),
        "codex_version": codex_version,
        "reader_prompt_version": READER_PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "reader_models": args.reader_models,
        "judge_models": args.judge_models,
        "architectures": args.architectures,
        "sample_ids": ids_by_stratum,
        "per_stratum": args.per_stratum,
        "repetitions": args.repetitions,
        "reasoning_effort": args.reasoning_effort,
        "timeout_seconds": args.timeout_seconds,
        "provider_retries": args.provider_retries,
        "execution_seed": args.execution_seed,
        "execution_order": "interleaved",
    }
    rows_by_key: dict[str, dict[str, Any]] = {}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        previous_manifest = previous.get("manifest", {})
        mismatches = {
            key: {"existing": previous_manifest.get(key), "requested": value}
            for key, value in protocol.items()
            if previous_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume manifest mismatch: {mismatches}")
        rows_by_key = {row["run_key"]: row for row in previous.get("rows", [])}

    expected_reader_calls = (
        len(instances)
        * len(args.reader_models)
        * len(args.architectures)
        * args.repetitions
    )
    reader_calls = 0
    reader_schedule = interleaved_product(
        range(args.repetitions),
        args.reader_models,
        list(STRATA),
        args.architectures,
        range(args.per_stratum),
        seed=args.execution_seed,
    )
    for repetition, reader_model, stratum, architecture, position in reader_schedule:
        instance_id = ids_by_stratum[stratum][position]
        run_key = "|".join(
            [
                stratum,
                instance_id,
                architecture,
                reader_model,
                str(repetition),
            ]
        )
        previous = rows_by_key.get(run_key)
        if previous and (previous.get("reader_ok") or not args.retry_errors):
            continue
        if args.max_reader_calls is not None and reader_calls >= args.max_reader_calls:
            break
        instance = instances[(stratum, instance_id)]
        notes = retrieve_notes(instance, architecture)
        prompt = render_reader_prompt(instance, notes)
        result = run_codex(
            args.codex_bin,
            reader_model,
            args.reasoning_effort,
            answer_schema,
            prompt,
            args.timeout_seconds,
            args.provider_retries,
        )
        response = result.get("response")
        predicted = (
            str(response.get("answer", "")).strip()
            if isinstance(response, dict)
            else ""
        )
        gold = str(instance["answer"])
        rows_by_key[run_key] = {
            "run_key": run_key,
            "stratum": stratum,
            "instance_id": instance_id,
            "num_hops": instance.get("num_hops"),
            "architecture": architecture,
            "reader_model": reader_model,
            "repetition": repetition,
            "question_sha256": hashlib.sha256(
                str(instance["question"]).encode()
            ).hexdigest(),
            "reader_ok": result["ok"],
            "predicted_answer": predicted,
            "gold_answer": gold,
            "exact_match": bool(predicted) and exact_match(predicted, [gold]),
            "substring_match": bool(predicted)
            and substring_match(predicted, [gold]),
            "token_f1": token_f1(predicted, [gold]) if predicted else 0.0,
            "context_words": len(notes.split())
            + sum(
                len(str(document.get("text", "")).split())
                for document in visible_documents(
                    instance, len(instance["turns"]) - 1
                )
            ),
            "reader_latency_seconds": result["latency_seconds"],
            "reader_tokens": result["tokens_used"],
            "reader_raw_output_sha256": result["raw_output_sha256"],
            "reader_stderr_tail": result["stderr_tail"],
            "judges": [],
        }
        reader_calls += 1
        save_payload(
            output,
            protocol,
            list(rows_by_key.values()),
            expected_reader_calls,
        )

    judge_calls = 0
    reader_keys = sorted(rows_by_key)
    judge_schedule = interleaved_product(
        args.judge_models,
        reader_keys,
        seed=args.execution_seed + 1,
    )
    for judge_model, run_key in judge_schedule:
        row = rows_by_key[run_key]
        if not row.get("reader_ok") or not row.get("predicted_answer"):
            continue
        existing_judge = next(
            (
                judge
                for judge in row.get("judges", [])
                if judge["model"] == judge_model
            ),
            None,
        )
        if existing_judge and (existing_judge.get("ok") or not args.retry_errors):
            continue
        if args.max_judge_calls is not None and judge_calls >= args.max_judge_calls:
            break
        stratum, instance_id = row["stratum"], row["instance_id"]
        instance = instances[(stratum, instance_id)]
        result = run_codex(
            args.codex_bin,
            judge_model,
            args.reasoning_effort,
            judge_schema,
            render_judge_prompt(
                str(instance["question"]),
                row["gold_answer"],
                row["predicted_answer"],
            ),
            args.timeout_seconds,
            args.provider_retries,
        )
        judges = [
            judge
            for judge in row.get("judges", [])
            if judge["model"] != judge_model
        ]
        judges.append(
            {
                "model": judge_model,
                "ok": result["ok"],
                "response": result.get("response"),
                "latency_seconds": result["latency_seconds"],
                "tokens_used": result["tokens_used"],
                "raw_output_sha256": result["raw_output_sha256"],
                "stderr_tail": result["stderr_tail"],
            }
        )
        row["judges"] = judges
        judge_calls += 1
        save_payload(
            output,
            protocol,
            list(rows_by_key.values()),
            expected_reader_calls,
        )

    save_payload(
        output,
        protocol,
        list(rows_by_key.values()),
        expected_reader_calls,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
