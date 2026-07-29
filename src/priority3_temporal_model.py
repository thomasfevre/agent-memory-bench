#!/usr/bin/env python3
"""Run resumable local-model extraction for the Priority 3 temporal corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENT_TYPES = {
    "assertion",
    "correction",
    "late_arrival",
    "duplicate",
    "expiration",
    "retraction",
    "low_confidence_contradiction",
    "shard_approval",
    "shard_rejection",
}

EXTRACTION_FIELDS = {
    "entity_key",
    "value",
    "asserted_at",
    "effective_from",
    "effective_until",
    "event_type",
    "target_event_id",
    "confidence",
    "source_id",
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entity_key": {"type": "string"},
        "value": {"type": "string"},
        "asserted_at": {"type": "string"},
        "effective_from": {"type": "string"},
        "effective_until": {"type": ["string", "null"]},
        "event_type": {"type": "string", "enum": sorted(EVENT_TYPES)},
        "target_event_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "source_id": {"type": "string"},
    },
    "required": sorted(EXTRACTION_FIELDS),
}

Extractor = Callable[..., dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_extraction(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EXTRACTION_FIELDS:
        raise ValueError("extraction fields do not match the public contract")
    for field in (
        "entity_key",
        "value",
        "asserted_at",
        "effective_from",
        "event_type",
        "source_id",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{field} must be a non-empty string")
    for field in ("effective_until", "target_event_id"):
        if value[field] is not None and not isinstance(value[field], str):
            raise ValueError(f"{field} must be a string or null")
    if value["event_type"] not in EVENT_TYPES:
        raise ValueError("event_type is not allowed")
    confidence = value["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be a number between zero and one")
    return value


def build_extraction_prompt(observation_text: str) -> str:
    return f"""Extract one temporal memory event from the observation.
Do not add facts that are absent. Preserve identifiers and ISO timestamps exactly.
Use null when effective_until or target_event_id is not stated.
Return exactly one JSON object with these fields:
entity_key, value, asserted_at, effective_from, effective_until, event_type,
target_event_id, confidence, source_id.

Allowed event_type values:
{", ".join(sorted(EVENT_TYPES))}

Observation:
{observation_text}
"""


def extract_observation_ollama(
    observation_text: str,
    *,
    model: str,
    seed: int,
    endpoint: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_extraction_prompt(observation_text),
            }
        ],
        "stream": False,
        "think": False,
        "format": EXTRACTION_SCHEMA,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": 4096,
            "num_predict": 384,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(
        request,
        timeout=timeout_seconds,
    ) as response:
        raw = json.load(response)
    latency_ms = (time.perf_counter() - started) * 1000
    content = raw.get("message", {}).get("content", "")
    parsed = validate_extraction(json.loads(content))
    return {
        "extraction": parsed,
        "metrics": {
            "latency_ms": latency_ms,
            "prompt_tokens": int(raw.get("prompt_eval_count", 0)),
            "output_tokens": int(raw.get("eval_count", 0)),
            "load_duration_ns": int(raw.get("load_duration", 0)),
            "prompt_eval_duration_ns": int(
                raw.get("prompt_eval_duration", 0)
            ),
            "eval_duration_ns": int(raw.get("eval_duration", 0)),
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_extraction_campaign(
    *,
    observations_path: Path,
    output_path: Path,
    model: str,
    repetitions: int,
    seed: int,
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_seconds: float = 180,
    extractor: Extractor = extract_observation_ollama,
    retry_errors: bool = False,
) -> dict[str, Any]:
    observations = load_jsonl(observations_path)
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result["corpus_sha256"] != sha256_file(observations_path):
            raise ValueError("checkpoint corpus hash does not match")
        if result["model"] != model or result["repetitions"] != repetitions:
            raise ValueError("checkpoint campaign settings do not match")
    else:
        result = {
            "schema_version": 1,
            "campaign": "priority3_temporal_extraction",
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "corpus_path": str(observations_path),
            "corpus_sha256": sha256_file(observations_path),
            "model": model,
            "endpoint": endpoint,
            "repetitions": repetitions,
            "seed": seed,
            "rows": [],
        }

    expected_rows = repetitions * len(observations)
    if (
        result.get("status") == "complete"
        and len(result["rows"]) == expected_rows
        and not retry_errors
    ):
        return result

    rows_by_key = {
        (row["repetition"], row["event_id"]): row for row in result["rows"]
    }
    for repetition in range(1, repetitions + 1):
        for observation_index, observation in enumerate(observations):
            key = (repetition, observation["id"])
            previous = rows_by_key.get(key)
            if previous and not (
                retry_errors and previous["status"] == "error"
            ):
                continue
            try:
                extracted = extractor(
                    observation["text"],
                    model=model,
                    seed=seed + observation_index,
                    endpoint=endpoint,
                    timeout_seconds=timeout_seconds,
                )
                row = {
                    "repetition": repetition,
                    "event_id": observation["id"],
                    "status": "success",
                    **extracted,
                }
            except Exception as error:  # retained as experimental evidence
                row = {
                    "repetition": repetition,
                    "event_id": observation["id"],
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            rows_by_key[key] = row
            result["rows"] = [
                rows_by_key[current_key]
                for current_key in sorted(rows_by_key)
            ]
            result["updated_at"] = utc_now()
            save_checkpoint(output_path, result)

    result["status"] = (
        "complete"
        if len(result["rows"]) == expected_rows
        else "incomplete"
    )
    result["successful_rows"] = sum(
        row["status"] == "success" for row in result["rows"]
    )
    result["error_rows"] = sum(
        row["status"] == "error" for row in result["rows"]
    )
    result["updated_at"] = utc_now()
    save_checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434/api/chat",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_extraction_campaign(
        observations_path=args.observations,
        output_path=args.output,
        model=args.model,
        repetitions=args.repetitions,
        seed=args.seed,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        retry_errors=args.retry_errors,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "successful_rows": result["successful_rows"],
                "error_rows": result["error_rows"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["error_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
