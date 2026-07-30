"""Local-only human review data contracts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


JUDGE_FIELDS = [
    "item_id",
    "score",
    "confidence",
    "time_seconds",
    "notes",
]
JUDGE_SCORES = (0.0, 0.3, 0.5, 0.7, 1.0)
SHARD_FIELDS = [
    "item_id",
    "decision",
    "scope",
    "injection",
    "confidence",
    "time_seconds",
    "notes",
]
SHARD_LABELS = ("approved", "rejected", "deferred")
SHARD_SCOPES = ("personal", "team", "task")
SHARD_INJECTIONS = ("always_on", "task_specific", "never")


def validate_numeric_field(
    item_id: str,
    field: str,
    value: Any,
    *,
    minimum: float,
    maximum: float | None = None,
) -> Any:
    if value == "":
        return value
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{item_id} {field} must be numeric") from error
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{item_id} {field} is outside the allowed range")
    if maximum is not None and number > maximum:
        raise ValueError(f"{item_id} {field} is outside the allowed range")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_review_state(
    campaign_id: str,
    label: str,
    pack_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    items = read_jsonl(pack_path)
    task_types = {row["task_type"] for row in items}
    if len(task_types) != 1:
        raise ValueError("review pack must contain exactly one task type")
    task_type = next(iter(task_types))
    fields = (
        JUDGE_FIELDS
        if task_type == "semantic_answer_judge"
        else SHARD_FIELDS
        if task_type == "context_shard"
        else None
    )
    if fields is None:
        raise ValueError(f"unsupported review task type: {task_type}")
    saved: dict[str, dict[str, str]] = {}
    if output_path.is_file():
        with output_path.open(newline="", encoding="utf-8") as handle:
            saved = {
                row["item_id"]: row
                for row in csv.DictReader(handle)
                if row.get("item_id")
            }
    annotations = {
        item["item_id"]: {
            field: saved.get(item["item_id"], {}).get(field, "")
            for field in fields
            if field != "item_id"
        }
        for item in items
    }
    completion_field = (
        "score" if task_type == "semantic_answer_judge" else "decision"
    )
    return {
        "campaign_id": campaign_id,
        "label": label,
        "task_type": task_type,
        "total_items": len(items),
        "completed_items": sum(
            bool(row[completion_field].strip())
            for row in annotations.values()
        ),
        "items": items,
        "annotations": annotations,
    }


def save_annotations(
    pack_path: Path,
    output_path: Path,
    annotations: dict[str, dict[str, Any]],
) -> None:
    rows = read_jsonl(pack_path)
    task_types = {row["task_type"] for row in rows}
    if len(task_types) != 1:
        raise ValueError("unsupported review pack")
    task_type = next(iter(task_types))
    if task_type == "semantic_answer_judge":
        fields = JUDGE_FIELDS
    elif task_type == "context_shard":
        fields = SHARD_FIELDS
    else:
        raise ValueError("unsupported review pack")
    known_ids = {row["item_id"] for row in rows}
    unknown_ids = sorted(set(annotations) - known_ids)
    if unknown_ids:
        raise ValueError(f"annotations contain unknown item IDs: {unknown_ids}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            item_id = item["item_id"]
            annotation = annotations.get(item_id, {})
            confidence = validate_numeric_field(
                item_id,
                "confidence",
                annotation.get("confidence", ""),
                minimum=0,
                maximum=1,
            )
            time_seconds = validate_numeric_field(
                item_id,
                "time_seconds",
                annotation.get("time_seconds", ""),
                minimum=0,
            )
            if task_type == "semantic_answer_judge":
                score = annotation.get("score", "")
                if score != "" and float(score) not in JUDGE_SCORES:
                    raise ValueError(
                        f"{item_id} score must be one of {JUDGE_SCORES}"
                    )
                output = {
                    "item_id": item_id,
                    "score": score,
                    "confidence": confidence,
                    "time_seconds": time_seconds,
                    "notes": annotation.get("notes", ""),
                }
            else:
                decision = annotation.get("decision", "")
                scope = annotation.get("scope", "")
                injection = annotation.get("injection", "")
                if decision != "" and decision not in SHARD_LABELS:
                    raise ValueError(
                        f"{item_id} decision must be one of {SHARD_LABELS}"
                    )
                if scope != "" and scope not in SHARD_SCOPES:
                    raise ValueError(
                        f"{item_id} scope must be one of {SHARD_SCOPES}"
                    )
                if injection != "" and injection not in SHARD_INJECTIONS:
                    raise ValueError(
                        f"{item_id} injection must be one of "
                        f"{SHARD_INJECTIONS}"
                    )
                output = {
                    "item_id": item_id,
                    "decision": decision,
                    "scope": scope,
                    "injection": injection,
                    "confidence": confidence,
                    "time_seconds": time_seconds,
                    "notes": annotation.get("notes", ""),
                }
            writer.writerow(output)
    temporary.replace(output_path)
