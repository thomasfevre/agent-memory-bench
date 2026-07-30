from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from human_review_app import load_review_state, save_annotations


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_semantic_review_exports_the_frozen_scorer_csv_contract(tmp_path):
    pack = tmp_path / "review-pack.jsonl"
    output = tmp_path / "annotator-owner.csv"
    write_jsonl(
        pack,
        [
            {
                "item_id": "judge-a",
                "task_type": "semantic_answer_judge",
                "score_options": [0.0, 0.3, 0.5, 0.7, 1.0],
            },
            {
                "item_id": "judge-b",
                "task_type": "semantic_answer_judge",
                "score_options": [0.0, 0.3, 0.5, 0.7, 1.0],
            },
        ],
    )

    save_annotations(
        pack,
        output,
        {
            "judge-a": {
                "score": 1.0,
                "confidence": 0.9,
                "time_seconds": 12,
                "notes": "Correct.",
            },
            "judge-b": {
                "score": 0.3,
                "confidence": 0.7,
                "time_seconds": 18,
                "notes": "",
            },
        },
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "item_id",
        "score",
        "confidence",
        "time_seconds",
        "notes",
    ]
    assert rows == [
        {
            "item_id": "judge-a",
            "score": "1.0",
            "confidence": "0.9",
            "time_seconds": "12",
            "notes": "Correct.",
        },
        {
            "item_id": "judge-b",
            "score": "0.3",
            "confidence": "0.7",
            "time_seconds": "18",
            "notes": "",
        },
    ]


def test_context_shard_review_exports_the_frozen_scorer_csv_contract(tmp_path):
    pack = tmp_path / "review-pack.jsonl"
    output = tmp_path / "annotator-owner.csv"
    write_jsonl(
        pack,
        [
            {
                "item_id": "shard-a",
                "task_type": "context_shard",
                "decision_options": ["approved", "rejected", "deferred"],
                "scope_options": ["personal", "team", "task"],
                "injection_options": [
                    "always_on",
                    "task_specific",
                    "never",
                ],
            }
        ],
    )

    save_annotations(
        pack,
        output,
        {
            "shard-a": {
                "decision": "approved",
                "scope": "personal",
                "injection": "task_specific",
                "confidence": 0.8,
                "time_seconds": 24,
                "notes": "Useful for my workflow.",
            }
        },
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "item_id",
        "decision",
        "scope",
        "injection",
        "confidence",
        "time_seconds",
        "notes",
    ]
    assert rows == [
        {
            "item_id": "shard-a",
            "decision": "approved",
            "scope": "personal",
            "injection": "task_specific",
            "confidence": "0.8",
            "time_seconds": "24",
            "notes": "Useful for my workflow.",
        }
    ]


def test_review_state_resumes_saved_progress_without_exposing_private_mapping(
    tmp_path,
):
    pack = tmp_path / "review-pack.jsonl"
    output = tmp_path / "annotator-owner.csv"
    write_jsonl(
        pack,
        [
            {
                "item_id": "judge-a",
                "task_type": "semantic_answer_judge",
                "question": "Question A?",
                "gold_answer": "Gold A",
                "predicted_answer": "Prediction A",
                "score_options": [0.0, 0.3, 0.5, 0.7, 1.0],
            },
            {
                "item_id": "judge-b",
                "task_type": "semantic_answer_judge",
                "question": "Question B?",
                "gold_answer": "Gold B",
                "predicted_answer": "Prediction B",
                "score_options": [0.0, 0.3, 0.5, 0.7, 1.0],
            },
        ],
    )
    save_annotations(
        pack,
        output,
        {
            "judge-a": {
                "score": 0.7,
                "confidence": 0.8,
                "time_seconds": 15,
                "notes": "",
            }
        },
    )

    state = load_review_state(
        campaign_id="memgym",
        label="Réponses MemGym",
        pack_path=pack,
        output_path=output,
    )

    assert state["campaign_id"] == "memgym"
    assert state["task_type"] == "semantic_answer_judge"
    assert state["total_items"] == 2
    assert state["completed_items"] == 1
    assert state["annotations"]["judge-a"]["score"] == "0.7"
    assert state["annotations"]["judge-b"]["score"] == ""
    assert "mapping" not in json.dumps(state).lower()


@pytest.mark.parametrize("confidence", [-0.1, 1.1, "not-a-number"])
def test_review_rejects_confidence_outside_zero_to_one(
    tmp_path,
    confidence,
):
    pack = tmp_path / "review-pack.jsonl"
    output = tmp_path / "annotator-owner.csv"
    write_jsonl(
        pack,
        [
            {
                "item_id": "judge-a",
                "task_type": "semantic_answer_judge",
            }
        ],
    )

    with pytest.raises(ValueError, match="confidence"):
        save_annotations(
            pack,
            output,
            {
                "judge-a": {
                    "score": 1.0,
                    "confidence": confidence,
                    "time_seconds": 12,
                }
            },
        )


@pytest.mark.parametrize("time_seconds", [-1, "not-a-number"])
def test_review_rejects_invalid_active_time(tmp_path, time_seconds):
    pack = tmp_path / "review-pack.jsonl"
    output = tmp_path / "annotator-owner.csv"
    write_jsonl(
        pack,
        [
            {
                "item_id": "shard-a",
                "task_type": "context_shard",
            }
        ],
    )

    with pytest.raises(ValueError, match="time_seconds"):
        save_annotations(
            pack,
            output,
            {
                "shard-a": {
                    "decision": "approved",
                    "scope": "personal",
                    "injection": "task_specific",
                    "confidence": 0.8,
                    "time_seconds": time_seconds,
                }
            },
        )
