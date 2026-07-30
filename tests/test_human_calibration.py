from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "human_calibration.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_cli(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_prepare_shards_blinds_reference_labels_and_ids(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    shards = tmp_path / "shards.jsonl"
    output = tmp_path / "pack"
    write_jsonl(
        corpus,
        [
            {"id": "d1", "timestamp": "2026-01-01", "text": "Use UTC."},
            {"id": "d2", "timestamp": "2026-01-02", "text": "UTC again."},
        ],
    )
    write_jsonl(
        shards,
        [
            {
                "id": "secret-shard",
                "text": "Always use UTC.",
                "source_ids": ["d1", "d2"],
                "review": "approved",
                "scope": "incidents",
            }
        ],
    )

    run_cli(
        "prepare-shards",
        "--shards",
        str(shards),
        "--corpus",
        str(corpus),
        "--output-dir",
        str(output),
        "--seed",
        "42",
    )

    public_row = json.loads((output / "review-pack.jsonl").read_text())
    mapping_row = json.loads((output / "private-mapping.jsonl").read_text())
    assert public_row["item_id"].startswith("shard-")
    assert "secret-shard" not in json.dumps(public_row)
    assert "review" not in public_row
    assert public_row["candidate_text"] == "Always use UTC."
    assert len(public_row["evidence"]) == 2
    assert mapping_row["source_key"] == "secret-shard"
    assert mapping_row["reference_label"] == "approved"
    assert mapping_row["reference_status"] == "synthetic"
    assert (output / "annotator-a.csv").exists()
    assert (output / "annotator-b.csv").exists()


def test_prepare_judge_pack_hides_model_judgments(tmp_path):
    result = tmp_path / "memgym.json"
    output = tmp_path / "pack"
    result.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "run_key": "secret-run",
                        "stratum": "3hop",
                        "architecture": "bm25_k2",
                        "reader_model": "reader-secret",
                        "question": "Question?",
                        "gold_answer": "Gold",
                        "predicted_answer": "Prediction",
                        "reader_ok": True,
                        "judges": [
                            {
                                "model": "judge-secret",
                                "ok": True,
                                "response": {"score": 0.7},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_cli(
        "prepare-judge",
        "--result",
        str(result),
        "--output-dir",
        str(output),
        "--sample-size",
        "1",
        "--seed",
        "42",
    )

    public_row = json.loads((output / "review-pack.jsonl").read_text())
    mapping_row = json.loads((output / "private-mapping.jsonl").read_text())
    public_text = json.dumps(public_row)
    assert public_row["item_id"].startswith("judge-")
    assert "secret-run" not in public_text
    assert "reader-secret" not in public_text
    assert "judge-secret" not in public_text
    assert "model_judge_scores" not in public_row
    assert "judges" not in public_row
    assert mapping_row["source_key"] == "secret-run"
    assert mapping_row["model_judge_scores"] == [0.7]


def test_prepare_judge_pack_balances_strata_before_taking_more_from_one(tmp_path):
    result = tmp_path / "memgym.json"
    output = tmp_path / "pack"
    rows = []
    for stratum in ("A", "B"):
        for index in range(1, 5):
            rows.append(
                {
                    "run_key": f"{stratum}{index}",
                    "stratum": stratum,
                    "architecture": "bm25_k2",
                    "reader_model": "reader",
                    "question": f"Question {stratum}{index}?",
                    "gold_answer": "Gold",
                    "predicted_answer": "Prediction",
                    "reader_ok": True,
                    "judges": [],
                }
            )
    result.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    run_cli(
        "prepare-judge",
        "--result",
        str(result),
        "--output-dir",
        str(output),
        "--sample-size",
        "2",
        "--seed",
        "42",
    )

    mapping = [
        json.loads(line)
        for line in (output / "private-mapping.jsonl").read_text().splitlines()
    ]
    assert {row["stratum"] for row in mapping} == {"A", "B"}


def test_prepare_judge_resolves_hashed_question_from_local_dataset(tmp_path):
    import hashlib

    result = tmp_path / "memgym.json"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output = tmp_path / "pack"
    question = "Which evidence supports the final answer?"
    question_sha256 = hashlib.sha256(question.encode()).hexdigest()
    write_jsonl(
        dataset_dir / "3hop_verified.jsonl",
        [{"instance_id": "i1", "question": question}],
    )
    result.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "run_key": "secret-run",
                        "stratum": "3hop",
                        "architecture": "bm25_k2",
                        "question_sha256": question_sha256,
                        "gold_answer": "Gold",
                        "predicted_answer": "Prediction",
                        "reader_ok": True,
                        "judges": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    run_cli(
        "prepare-judge",
        "--result",
        str(result),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output),
        "--sample-size",
        "1",
        "--seed",
        "42",
    )

    public_row = json.loads((output / "review-pack.jsonl").read_text())
    assert public_row["question"] == question


def test_score_shard_review_reports_agreement_accuracy_and_time(tmp_path):
    pack = tmp_path / "pack.jsonl"
    mapping = tmp_path / "mapping.jsonl"
    annotator_a = tmp_path / "a.csv"
    annotator_b = tmp_path / "b.csv"
    output = tmp_path / "agreement.json"
    item_ids = ["i1", "i2", "i3", "i4"]
    references = ["approved", "approved", "rejected", "deferred"]
    write_jsonl(
        pack,
        [
            {"item_id": item_id, "task_type": "context_shard"}
            for item_id in item_ids
        ],
    )
    write_jsonl(
        mapping,
        [
            {
                "item_id": item_id,
                "source_key": f"s{index}",
                "reference_label": reference,
                "reference_status": "synthetic",
            }
            for index, (item_id, reference) in enumerate(
                zip(item_ids, references, strict=True)
            )
        ],
    )

    def write_labels(path: Path, labels: list[str], times: list[float]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "item_id",
                    "decision",
                    "scope",
                    "injection",
                    "confidence",
                    "time_seconds",
                    "notes",
                ],
            )
            writer.writeheader()
            for item_id, label, elapsed in zip(
                item_ids, labels, times, strict=True
            ):
                writer.writerow(
                    {
                        "item_id": item_id,
                        "decision": label,
                        "scope": "team",
                        "injection": "task_specific",
                        "confidence": "0.8",
                        "time_seconds": elapsed,
                        "notes": "",
                    }
                )

    write_labels(annotator_a, references, [10, 20, 30, 40])
    write_labels(
        annotator_b,
        ["approved", "rejected", "rejected", "deferred"],
        [12, 22, 32, 42],
    )

    run_cli(
        "score",
        "--pack",
        str(pack),
        "--mapping",
        str(mapping),
        "--annotator-a",
        str(annotator_a),
        "--annotator-b",
        str(annotator_b),
        "--output",
        str(output),
    )

    report = json.loads(output.read_text())
    assert report["coverage"]["paired_items"] == 4
    assert report["agreement"]["exact"] == 0.75
    assert report["agreement"]["cohen_kappa"] == 0.636364
    assert report["reference"]["annotator_a_accuracy"] == 1.0
    assert report["reference"]["annotator_b_accuracy"] == 0.75
    assert report["review_time_seconds"]["annotator_a_median"] == 25.0
    assert report["review_time_seconds"]["annotator_b_median"] == 27.0


def test_score_rejects_semantic_scores_outside_the_frozen_scale(tmp_path):
    pack = tmp_path / "pack.jsonl"
    mapping = tmp_path / "mapping.jsonl"
    annotator_a = tmp_path / "a.csv"
    annotator_b = tmp_path / "b.csv"
    output = tmp_path / "agreement.json"
    write_jsonl(
        pack,
        [{"item_id": "i1", "task_type": "semantic_answer_judge"}],
    )
    write_jsonl(
        mapping,
        [{"item_id": "i1", "source_key": "s1", "model_judge_scores": [0.7]}],
    )
    for path, score in ((annotator_a, "0.8"), (annotator_b, "0.7")):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "item_id",
                    "score",
                    "confidence",
                    "time_seconds",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerow({"item_id": "i1", "score": score})

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "score",
            "--pack",
            str(pack),
            "--mapping",
            str(mapping),
            "--annotator-a",
            str(annotator_a),
            "--annotator-b",
            str(annotator_b),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "frozen semantic score scale" in completed.stderr
    assert not output.exists()


def test_score_rejects_context_shard_decisions_outside_the_frozen_labels(tmp_path):
    pack = tmp_path / "pack.jsonl"
    mapping = tmp_path / "mapping.jsonl"
    annotator_a = tmp_path / "a.csv"
    annotator_b = tmp_path / "b.csv"
    output = tmp_path / "agreement.json"
    write_jsonl(
        pack,
        [{"item_id": "i1", "task_type": "context_shard"}],
    )
    write_jsonl(
        mapping,
        [
            {
                "item_id": "i1",
                "source_key": "s1",
                "reference_label": "approved",
                "reference_status": "synthetic",
            }
        ],
    )
    for path, decision in (
        (annotator_a, "accept"),
        (annotator_b, "approved"),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "item_id",
                    "decision",
                    "scope",
                    "injection",
                    "confidence",
                    "time_seconds",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerow({"item_id": "i1", "decision": decision})

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "score",
            "--pack",
            str(pack),
            "--mapping",
            str(mapping),
            "--annotator-a",
            str(annotator_a),
            "--annotator-b",
            str(annotator_b),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "frozen Context Shard decision labels" in completed.stderr
    assert not output.exists()
