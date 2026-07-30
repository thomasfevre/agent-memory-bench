#!/usr/bin/env python3
"""Prepare blinded human-review packs and score independent annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SHARD_FIELDS = [
    "item_id",
    "decision",
    "scope",
    "injection",
    "confidence",
    "time_seconds",
    "notes",
]
JUDGE_FIELDS = [
    "item_id",
    "score",
    "confidence",
    "time_seconds",
    "notes",
]
SHARD_LABELS = ("approved", "rejected", "deferred")
JUDGE_SCORES = (0.0, 0.3, 0.5, 0.7, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    shards = commands.add_parser("prepare-shards")
    shards.add_argument("--shards", type=Path, required=True)
    shards.add_argument("--corpus", type=Path, required=True)
    shards.add_argument("--output-dir", type=Path, required=True)
    shards.add_argument("--seed", type=int, default=20260729)

    judge = commands.add_parser("prepare-judge")
    judge.add_argument("--result", type=Path, required=True)
    judge.add_argument("--dataset-dir", type=Path)
    judge.add_argument("--output-dir", type=Path, required=True)
    judge.add_argument("--sample-size", type=int, default=40)
    judge.add_argument("--seed", type=int, default=20260729)

    owner_mini = commands.add_parser("prepare-owner-mini")
    owner_mini.add_argument("--memgym-pack", type=Path, required=True)
    owner_mini.add_argument("--memgym-mapping", type=Path, required=True)
    owner_mini.add_argument("--shards-pack", type=Path, required=True)
    owner_mini.add_argument("--shards-mapping", type=Path, required=True)
    owner_mini.add_argument("--output-dir", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--pack", type=Path, required=True)
    score.add_argument("--mapping", type=Path, required=True)
    score.add_argument("--annotator-a", type=Path, required=True)
    score.add_argument("--annotator-b", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    score_single = commands.add_parser("score-single")
    score_single.add_argument("--pack", type=Path, required=True)
    score_single.add_argument("--mapping", type=Path, required=True)
    score_single.add_argument("--annotator", type=Path, required=True)
    score_single.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blind_id(prefix: str, source_key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{source_key}".encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def random_order_key(source_key: str, seed: int) -> str:
    return hashlib.sha256(f"order|{seed}|{source_key}".encode()).hexdigest()


def write_annotation_templates(
    output_dir: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    for suffix in ("a", "b"):
        with (output_dir / f"annotator-{suffix}.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({"item_id": row["item_id"]})


def write_owner_template(
    output_dir: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path = output_dir / "annotator-owner.csv"
    selected_ids = [row["item_id"] for row in rows]
    if path.exists():
        existing = read_annotations(path)
        if list(existing) != selected_ids:
            raise ValueError(
                f"existing owner annotations do not match selected items: {path}"
            )
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_id in selected_ids:
            writer.writerow({"item_id": item_id})


def model_score_band(score: float) -> int:
    if score <= 0.2:
        return 0
    if score <= 0.4:
        return 1
    if score <= 0.6:
        return 2
    if score <= 0.8:
        return 3
    return 4


def select_diverse_memgym(
    pack_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
) -> list[str]:
    pack = {row["item_id"]: row for row in pack_rows}
    architectures = {str(row.get("architecture")) for row in mapping_rows}
    strata = {str(row.get("stratum")) for row in mapping_rows}
    required_architectures = min(4, len(architectures))
    required_strata = min(3, len(strata))
    maximum_per_stratum = math.ceil(5 / max(required_strata, 1))
    candidates: list[tuple[int, tuple[str, ...]]] = []
    for combination in itertools.combinations(mapping_rows, 5):
        scores = [
            float(row["model_judge_scores"][0])
            for row in combination
            if row.get("model_judge_scores")
        ]
        if len(scores) != 5 or len({model_score_band(score) for score in scores}) < 5:
            continue
        if (
            len({str(row.get("architecture")) for row in combination})
            < required_architectures
        ):
            continue
        stratum_counts = Counter(str(row.get("stratum")) for row in combination)
        if len(stratum_counts) < required_strata:
            continue
        if max(stratum_counts.values()) > maximum_per_stratum:
            continue
        item_ids = tuple(row["item_id"] for row in combination)
        reading_cost = sum(
            len(str(pack[item_id].get(field, "")))
            for item_id in item_ids
            for field in ("question", "gold_answer", "predicted_answer")
        )
        candidates.append((reading_cost, tuple(sorted(item_ids))))
    if not candidates:
        raise ValueError("cannot build a diverse five-item MemGym sample")
    selected = set(min(candidates)[1])
    return [row["item_id"] for row in pack_rows if row["item_id"] in selected]


def select_diverse_shards(
    pack_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
) -> list[str]:
    pack = {row["item_id"]: row for row in pack_rows}
    scopes = {str(row.get("original_scope")) for row in mapping_rows}
    required_scopes = min(5, len(scopes))
    available_evidence_counts = {
        len(pack[row["item_id"]].get("evidence", [])) for row in mapping_rows
    }
    required_evidence_counts = available_evidence_counts & {1, 2, 3}
    candidates: list[tuple[int, tuple[str, ...]]] = []
    for combination in itertools.combinations(mapping_rows, 5):
        label_counts = Counter(
            str(row.get("reference_label")) for row in combination
        )
        if set(label_counts) != set(SHARD_LABELS):
            continue
        if max(label_counts.values()) > 2:
            continue
        if (
            len({str(row.get("original_scope")) for row in combination})
            < required_scopes
        ):
            continue
        evidence_counts = {
            len(pack[row["item_id"]].get("evidence", []))
            for row in combination
        }
        if not required_evidence_counts.issubset(evidence_counts):
            continue
        item_ids = tuple(row["item_id"] for row in combination)
        reading_cost = sum(
            len(str(pack[item_id].get("candidate_text", "")))
            + sum(
                len(str(evidence.get("text", "")))
                for evidence in pack[item_id].get("evidence", [])
            )
            for item_id in item_ids
        )
        candidates.append((reading_cost, tuple(sorted(item_ids))))
    if not candidates:
        raise ValueError("cannot build a diverse five-item Context Shard sample")
    selected = set(min(candidates)[1])
    return [row["item_id"] for row in pack_rows if row["item_id"] in selected]


def write_mini_campaign(
    output_dir: Path,
    pack_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    selected_ids: list[str],
    fields: list[str],
) -> dict[str, Any]:
    selected = set(selected_ids)
    public_subset = [
        row for row in pack_rows if row["item_id"] in selected
    ]
    mapping_subset = [
        row for row in mapping_rows if row["item_id"] in selected
    ]
    if len(public_subset) != 5 or len(mapping_subset) != 5:
        raise ValueError("mini campaign must contain five aligned items")
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "review-pack.jsonl"
    mapping_path = output_dir / "private-mapping.jsonl"
    write_jsonl(pack_path, public_subset)
    write_jsonl(mapping_path, mapping_subset)
    write_owner_template(output_dir, public_subset, fields)
    return {
        "items": len(public_subset),
        "pack_sha256": sha256_file(pack_path),
        "mapping_sha256": sha256_file(mapping_path),
    }


def prepare_owner_mini(args: argparse.Namespace) -> None:
    memgym_pack = read_jsonl(args.memgym_pack)
    memgym_mapping = read_jsonl(args.memgym_mapping)
    shards_pack = read_jsonl(args.shards_pack)
    shards_mapping = read_jsonl(args.shards_mapping)
    memgym_ids = select_diverse_memgym(memgym_pack, memgym_mapping)
    shard_ids = select_diverse_shards(shards_pack, shards_mapping)
    memgym_manifest = write_mini_campaign(
        args.output_dir / "memgym",
        memgym_pack,
        memgym_mapping,
        memgym_ids,
        JUDGE_FIELDS,
    )
    shards_manifest = write_mini_campaign(
        args.output_dir / "context-shards",
        shards_pack,
        shards_mapping,
        shard_ids,
        SHARD_FIELDS,
    )
    write_json(
        args.output_dir / "manifest.json",
        {
            "protocol": "owner-mini-calibration-v1",
            "items": 10,
            "claim_boundary": (
                "Diverse exploratory owner sample, not statistically "
                "representative."
            ),
            "selection": {
                "memgym": (
                    "Five model-score bands, four architectures, three "
                    "reasoning-depth strata, then minimum reading length."
                ),
                "context_shards": (
                    "All three reference decisions, one-to-three evidence "
                    "items, five scopes, then minimum reading length."
                ),
            },
            "sources": {
                "memgym_pack_sha256": sha256_file(args.memgym_pack),
                "memgym_mapping_sha256": sha256_file(args.memgym_mapping),
                "shards_pack_sha256": sha256_file(args.shards_pack),
                "shards_mapping_sha256": sha256_file(args.shards_mapping),
            },
            "campaigns": {
                "memgym": memgym_manifest,
                "context_shards": shards_manifest,
            },
            "blinded": True,
            "required_annotators": 1,
        },
    )


def prepare_shards(args: argparse.Namespace) -> None:
    corpus = {row["id"]: row for row in read_jsonl(args.corpus)}
    shards = sorted(
        read_jsonl(args.shards),
        key=lambda row: random_order_key(row["id"], args.seed),
    )
    public_rows = []
    mapping_rows = []
    for shard in shards:
        item_id = blind_id("shard", shard["id"], args.seed)
        evidence = []
        for index, source_id in enumerate(shard.get("source_ids", []), start=1):
            source = corpus[source_id]
            evidence.append(
                {
                    "source_label": f"Source {index}",
                    "timestamp": source.get("timestamp"),
                    "text": source["text"],
                }
            )
        public_rows.append(
            {
                "item_id": item_id,
                "task_type": "context_shard",
                "candidate_text": shard["text"],
                "evidence": evidence,
                "decision_options": list(SHARD_LABELS),
                "scope_options": ["personal", "team", "task"],
                "injection_options": ["always_on", "task_specific", "never"],
            }
        )
        mapping_rows.append(
            {
                "item_id": item_id,
                "source_key": shard["id"],
                "reference_label": shard.get("review"),
                "reference_status": "synthetic",
                "original_scope": shard.get("scope"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = args.output_dir / "review-pack.jsonl"
    mapping_path = args.output_dir / "private-mapping.jsonl"
    write_jsonl(pack_path, public_rows)
    write_jsonl(mapping_path, mapping_rows)
    write_annotation_templates(args.output_dir, public_rows, SHARD_FIELDS)
    write_json(
        args.output_dir / "manifest.json",
        {
            "protocol": "context-shard-human-review-v1",
            "seed": args.seed,
            "items": len(public_rows),
            "task_type": "context_shard",
            "reference_status": "synthetic",
            "pack_sha256": sha256_file(pack_path),
            "mapping_sha256": sha256_file(mapping_path),
            "required_annotators": 2,
            "blinded": True,
        },
    )


def stratified_judge_sample(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("reader_ok")
        and row.get("predicted_answer") is not None
        and row.get("gold_answer") is not None
    ]
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[(str(row.get("stratum")), str(row.get("architecture")))].append(
            row
        )
    for group_rows in groups.values():
        group_rows.sort(
            key=lambda row: (
                random_order_key(row["run_key"], seed),
                row["run_key"],
            )
        )
    group_keys = sorted(
        groups,
        key=lambda key: random_order_key("|".join(key), seed),
    )
    selected = []
    round_index = 0
    while len(selected) < sample_size:
        added = False
        for group_key in group_keys:
            group_rows = groups[group_key]
            if round_index < len(group_rows):
                selected.append(group_rows[round_index])
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
        round_index += 1
    return selected


def load_questions_by_sha256(dataset_dir: Path) -> dict[str, str]:
    questions: dict[str, str] = {}
    for path in sorted(dataset_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            question = row.get("question")
            if not isinstance(question, str) or not question:
                continue
            digest = hashlib.sha256(question.encode()).hexdigest()
            existing = questions.get(digest)
            if existing is not None and existing != question:
                raise ValueError(f"question hash collision: {digest}")
            questions[digest] = question
    return questions


def prepare_judge(args: argparse.Namespace) -> None:
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    questions_by_sha256 = (
        load_questions_by_sha256(args.dataset_dir)
        if args.dataset_dir is not None
        else {}
    )
    selected = stratified_judge_sample(
        payload.get("rows", []),
        args.sample_size,
        args.seed,
    )
    public_rows = []
    mapping_rows = []
    for row in selected:
        item_id = blind_id("judge", row["run_key"], args.seed)
        question = row.get("question")
        if not question:
            question_digest = row.get("question_sha256")
            question = questions_by_sha256.get(str(question_digest))
        if not question:
            raise ValueError(
                "question text missing from result and unresolved from "
                f"dataset for run_key={row['run_key']}"
            )
        public_rows.append(
            {
                "item_id": item_id,
                "task_type": "semantic_answer_judge",
                "question": question,
                "gold_answer": row["gold_answer"],
                "predicted_answer": row["predicted_answer"],
                "score_options": list(JUDGE_SCORES),
            }
        )
        model_scores = [
            float(judge["response"]["score"])
            for judge in row.get("judges", [])
            if judge.get("ok")
            and isinstance(judge.get("response"), dict)
            and isinstance(judge["response"].get("score"), (int, float))
        ]
        mapping_rows.append(
            {
                "item_id": item_id,
                "source_key": row["run_key"],
                "stratum": row.get("stratum"),
                "architecture": row.get("architecture"),
                "model_judge_scores": model_scores,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = args.output_dir / "review-pack.jsonl"
    mapping_path = args.output_dir / "private-mapping.jsonl"
    write_jsonl(pack_path, public_rows)
    write_jsonl(mapping_path, mapping_rows)
    write_annotation_templates(args.output_dir, public_rows, JUDGE_FIELDS)
    write_json(
        args.output_dir / "manifest.json",
        {
            "protocol": "semantic-judge-human-calibration-v1",
            "seed": args.seed,
            "items": len(public_rows),
            "requested_sample_size": args.sample_size,
            "task_type": "semantic_answer_judge",
            "pack_sha256": sha256_file(pack_path),
            "mapping_sha256": sha256_file(mapping_path),
            "source_result_sha256": sha256_file(args.result),
            "question_dataset_sha256": (
                {
                    path.name: sha256_file(path)
                    for path in sorted(args.dataset_dir.glob("*.jsonl"))
                }
                if args.dataset_dir is not None
                else {}
            ),
            "required_annotators": 2,
            "blinded": True,
        },
    )


def read_annotations(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["item_id"]: row
            for row in csv.DictReader(handle)
            if row.get("item_id")
        }


def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if not first:
        return None
    labels = sorted(set(first) | set(second))
    observed = statistics.fmean(a == b for a, b in zip(first, second, strict=True))
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[label] * second_counts[label] for label in labels
    ) / (len(first) ** 2)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def pearson(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2:
        return None
    mean_first = statistics.fmean(first)
    mean_second = statistics.fmean(second)
    numerator = sum(
        (a - mean_first) * (b - mean_second)
        for a, b in zip(first, second, strict=True)
    )
    first_scale = math.sqrt(sum((a - mean_first) ** 2 for a in first))
    second_scale = math.sqrt(sum((b - mean_second) ** 2 for b in second))
    if math.isclose(first_scale, 0.0) or math.isclose(second_scale, 0.0):
        return None
    return numerator / (first_scale * second_scale)


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def numeric_values(
    rows: dict[str, dict[str, str]],
    item_ids: list[str],
    field: str,
) -> list[float]:
    return [
        float(rows[item_id][field])
        for item_id in item_ids
        if rows[item_id].get(field, "").strip()
    ]


def score_shards(
    item_ids: list[str],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels_first = [first[item_id]["decision"] for item_id in item_ids]
    labels_second = [second[item_id]["decision"] for item_id in item_ids]
    references = [
        mapping[item_id].get("reference_label")
        for item_id in item_ids
        if mapping.get(item_id, {}).get("reference_label") in SHARD_LABELS
    ]
    reference_ids = [
        item_id
        for item_id in item_ids
        if mapping.get(item_id, {}).get("reference_label") in SHARD_LABELS
    ]
    first_times = numeric_values(first, item_ids, "time_seconds")
    second_times = numeric_values(second, item_ids, "time_seconds")
    return {
        "agreement": {
            "exact": rounded(
                statistics.fmean(
                    a == b
                    for a, b in zip(labels_first, labels_second, strict=True)
                )
            ),
            "cohen_kappa": rounded(cohen_kappa(labels_first, labels_second)),
        },
        "reference": {
            "status": sorted(
                {
                    mapping[item_id].get("reference_status")
                    for item_id in reference_ids
                    if mapping[item_id].get("reference_status")
                }
            ),
            "items": len(references),
            "annotator_a_accuracy": rounded(
                statistics.fmean(
                    first[item_id]["decision"]
                    == mapping[item_id]["reference_label"]
                    for item_id in reference_ids
                )
            )
            if reference_ids
            else None,
            "annotator_b_accuracy": rounded(
                statistics.fmean(
                    second[item_id]["decision"]
                    == mapping[item_id]["reference_label"]
                    for item_id in reference_ids
                )
            )
            if reference_ids
            else None,
        },
        "review_time_seconds": {
            "annotator_a_median": rounded(statistics.median(first_times))
            if first_times
            else None,
            "annotator_b_median": rounded(statistics.median(second_times))
            if second_times
            else None,
        },
    }


def score_judges(
    item_ids: list[str],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scores_first = [float(first[item_id]["score"]) for item_id in item_ids]
    scores_second = [float(second[item_id]["score"]) for item_id in item_ids]
    model_rows = [
        (
            item_id,
            statistics.fmean(mapping[item_id]["model_judge_scores"]),
        )
        for item_id in item_ids
        if mapping.get(item_id, {}).get("model_judge_scores")
    ]
    model_scores = [score for _, score in model_rows]
    human_consensus = [
        statistics.fmean(
            [
                float(first[item_id]["score"]),
                float(second[item_id]["score"]),
            ]
        )
        for item_id, _ in model_rows
    ]
    return {
        "agreement": {
            "exact": rounded(
                statistics.fmean(
                    math.isclose(a, b)
                    for a, b in zip(scores_first, scores_second, strict=True)
                )
            ),
            "mean_absolute_difference": rounded(
                statistics.fmean(
                    abs(a - b)
                    for a, b in zip(scores_first, scores_second, strict=True)
                )
            ),
            "pearson": rounded(pearson(scores_first, scores_second)),
        },
        "model_judge": {
            "items": len(model_scores),
            "mean_absolute_error_to_human_consensus": rounded(
                statistics.fmean(
                    abs(model - human)
                    for model, human in zip(
                        model_scores,
                        human_consensus,
                        strict=True,
                    )
                )
            )
            if model_scores
            else None,
            "pearson_to_human_consensus": rounded(
                pearson(model_scores, human_consensus)
            ),
        },
        "review_time_seconds": {
            "annotator_a_median": rounded(
                statistics.median(numeric_values(first, item_ids, "time_seconds"))
            )
            if numeric_values(first, item_ids, "time_seconds")
            else None,
            "annotator_b_median": rounded(
                statistics.median(numeric_values(second, item_ids, "time_seconds"))
            )
            if numeric_values(second, item_ids, "time_seconds")
            else None,
        },
    }


def score(args: argparse.Namespace) -> None:
    pack_rows = read_jsonl(args.pack)
    mapping = {
        row["item_id"]: row for row in read_jsonl(args.mapping)
    }
    first = read_annotations(args.annotator_a)
    second = read_annotations(args.annotator_b)
    pack_ids = [row["item_id"] for row in pack_rows]
    paired = [
        item_id
        for item_id in pack_ids
        if item_id in first
        and item_id in second
        and (
            first[item_id].get("decision", "").strip()
            or first[item_id].get("score", "").strip()
        )
        and (
            second[item_id].get("decision", "").strip()
            or second[item_id].get("score", "").strip()
        )
    ]
    if not paired:
        raise ValueError("no independently completed item pair")
    task_types = {row["task_type"] for row in pack_rows}
    if len(task_types) != 1:
        raise ValueError("review pack must contain exactly one task type")
    task_type = next(iter(task_types))
    if task_type == "context_shard":
        invalid_labels = [
            (item_id, annotator, row[item_id]["decision"])
            for item_id in paired
            for annotator, row in (("a", first), ("b", second))
            if row[item_id]["decision"] not in SHARD_LABELS
        ]
        if invalid_labels:
            raise ValueError(
                "annotation is outside the frozen Context Shard decision "
                f"labels {SHARD_LABELS}: {invalid_labels}"
            )
        metrics = score_shards(paired, first, second, mapping)
    elif task_type == "semantic_answer_judge":
        invalid_scores = [
            (item_id, annotator, row[item_id]["score"])
            for item_id in paired
            for annotator, row in (("a", first), ("b", second))
            if float(row[item_id]["score"]) not in JUDGE_SCORES
        ]
        if invalid_scores:
            raise ValueError(
                "annotation is outside the frozen semantic score scale "
                f"{JUDGE_SCORES}: {invalid_scores}"
            )
        metrics = score_judges(paired, first, second, mapping)
    else:
        raise ValueError(f"unsupported task type: {task_type}")
    write_json(
        args.output,
        {
            "protocol": "human-calibration-agreement-v1",
            "task_type": task_type,
            "coverage": {
                "pack_items": len(pack_ids),
                "paired_items": len(paired),
                "paired_fraction": rounded(len(paired) / len(pack_ids)),
            },
            "inputs": {
                "pack_sha256": sha256_file(args.pack),
                "mapping_sha256": sha256_file(args.mapping),
                "annotator_a_sha256": sha256_file(args.annotator_a),
                "annotator_b_sha256": sha256_file(args.annotator_b),
            },
            **metrics,
        },
    )


def score_single(args: argparse.Namespace) -> None:
    pack_rows = read_jsonl(args.pack)
    mapping = {
        row["item_id"]: row for row in read_jsonl(args.mapping)
    }
    annotation = read_annotations(args.annotator)
    pack_ids = [row["item_id"] for row in pack_rows]
    annotated = [
        item_id
        for item_id in pack_ids
        if item_id in annotation
        and (
            annotation[item_id].get("decision", "").strip()
            or annotation[item_id].get("score", "").strip()
        )
    ]
    if not annotated:
        raise ValueError("no completed single-review item")
    task_types = {row["task_type"] for row in pack_rows}
    if len(task_types) != 1:
        raise ValueError("review pack must contain exactly one task type")
    task_type = next(iter(task_types))
    review_times = numeric_values(
        annotation,
        annotated,
        "time_seconds",
    )
    if task_type == "semantic_answer_judge":
        invalid_scores = [
            (item_id, annotation[item_id]["score"])
            for item_id in annotated
            if float(annotation[item_id]["score"]) not in JUDGE_SCORES
        ]
        if invalid_scores:
            raise ValueError(
                "annotation is outside the frozen semantic score scale "
                f"{JUDGE_SCORES}: {invalid_scores}"
            )
        model_rows = [
            (
                item_id,
                statistics.fmean(mapping[item_id]["model_judge_scores"]),
            )
            for item_id in annotated
            if mapping.get(item_id, {}).get("model_judge_scores")
        ]
        model_scores = [value for _, value in model_rows]
        human_scores = [
            float(annotation[item_id]["score"]) for item_id, _ in model_rows
        ]
        metrics = {
            "claim_boundary": (
                "Alignment with one owner-reviewer, not human consensus."
            ),
            "model_judge": {
                "items": len(model_scores),
                "mean_absolute_error_to_human": rounded(
                    statistics.fmean(
                        abs(model - human)
                        for model, human in zip(
                            model_scores,
                            human_scores,
                            strict=True,
                        )
                    )
                )
                if model_scores
                else None,
                "pearson_to_human": rounded(
                    pearson(model_scores, human_scores)
                ),
            },
        }
    elif task_type == "context_shard":
        invalid_labels = [
            (item_id, annotation[item_id]["decision"])
            for item_id in annotated
            if annotation[item_id]["decision"] not in SHARD_LABELS
        ]
        if invalid_labels:
            raise ValueError(
                "annotation is outside the frozen Context Shard decision "
                f"labels {SHARD_LABELS}: {invalid_labels}"
            )
        reference_ids = [
            item_id
            for item_id in annotated
            if mapping.get(item_id, {}).get("reference_label")
            in SHARD_LABELS
        ]
        metrics = {
            "claim_boundary": (
                "One owner's promotion policy, not team or population "
                "consensus."
            ),
            "decisions": {
                label: sum(
                    annotation[item_id]["decision"] == label
                    for item_id in annotated
                )
                for label in SHARD_LABELS
            },
            "reference": {
                "status": sorted(
                    {
                        mapping[item_id].get("reference_status")
                        for item_id in reference_ids
                        if mapping[item_id].get("reference_status")
                    }
                ),
                "items": len(reference_ids),
                "owner_accuracy": rounded(
                    statistics.fmean(
                        annotation[item_id]["decision"]
                        == mapping[item_id]["reference_label"]
                        for item_id in reference_ids
                    )
                )
                if reference_ids
                else None,
            },
        }
    else:
        raise ValueError(f"unsupported task type: {task_type}")
    write_json(
        args.output,
        {
            "protocol": "human-calibration-single-v1",
            "task_type": task_type,
            "coverage": {
                "pack_items": len(pack_ids),
                "annotated_items": len(annotated),
                "annotated_fraction": rounded(
                    len(annotated) / len(pack_ids)
                ),
            },
            "review_time_seconds": {
                "median": rounded(statistics.median(review_times))
                if review_times
                else None,
            },
            "inputs": {
                "pack_sha256": sha256_file(args.pack),
                "mapping_sha256": sha256_file(args.mapping),
                "annotator_sha256": sha256_file(args.annotator),
            },
            **metrics,
        },
    )


def main() -> None:
    args = parse_args()
    if args.command == "prepare-shards":
        prepare_shards(args)
    elif args.command == "prepare-judge":
        prepare_judge(args)
    elif args.command == "prepare-owner-mini":
        prepare_owner_mini(args)
    elif args.command == "score":
        score(args)
    elif args.command == "score-single":
        score_single(args)


if __name__ == "__main__":
    main()
