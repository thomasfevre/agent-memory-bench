#!/usr/bin/env python3
"""Validate the public registry without adding a runtime JSON-schema dependency."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "published" / "registry.json"
MANIFEST = ROOT / "results" / "published" / "raw-evidence-manifest.json"

RUN_KEYS = {
    "id",
    "date",
    "phase",
    "dataset",
    "task",
    "method",
    "evidence_level",
    "sample",
    "repetitions",
    "metrics",
    "conclusion",
    "limitation",
    "evidence_files",
}
PHASES = {"implementation", "ingestion", "retrieval", "generation", "agentic", "durability"}
LEVELS = {"controlled", "official-data", "smoke", "timeout", "not-reproduced"}


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []

    if registry.get("schema_version") != 1:
        errors.append("registry.schema_version must be 1")
    if len(registry.get("workflow", [])) < 5:
        errors.append("workflow must describe at least five boundaries")

    manifest_paths = {item["path"] for item in manifest.get("artifacts", [])}
    ids: set[str] = set()
    for index, run in enumerate(registry.get("runs", [])):
        prefix = f"runs[{index}]"
        missing = RUN_KEYS - set(run)
        if missing:
            errors.append(f"{prefix} missing {sorted(missing)}")
        run_id = run.get("id")
        if run_id in ids:
            errors.append(f"duplicate run id {run_id}")
        ids.add(run_id)
        if run.get("phase") not in PHASES:
            errors.append(f"{prefix} has invalid phase")
        if run.get("evidence_level") not in LEVELS:
            errors.append(f"{prefix} has invalid evidence level")
        if not isinstance(run.get("repetitions"), int) or run.get("repetitions", -1) < 0:
            errors.append(f"{prefix} repetitions must be a non-negative integer")
        if not run.get("metrics"):
            errors.append(f"{prefix} must expose at least one metric")
        for evidence in run.get("evidence_files", []):
            if evidence not in manifest_paths:
                errors.append(f"{prefix} references missing evidence {evidence}")

    if not registry.get("findings"):
        errors.append("findings must not be empty")
    if not registry.get("limitations"):
        errors.append("limitations must not be empty")

    if errors:
        print("\n".join(errors))
        return 1
    print(f"validated {len(ids)} public run records and {len(manifest_paths)} raw artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

