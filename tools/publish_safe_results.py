#!/usr/bin/env python3
"""Publish a small, reviewed set of synthetic benchmark artifacts.

Raw results are ignored by default. This explicit allowlist is the privacy and
licensing boundary: adding a file here means it has been reviewed as synthetic
or locally generated and safe to commit.
"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DESTINATION = RESULTS / "published" / "raw"

SAFE_RESULTS = (
    "COMMON-RETRIEVAL-COMPARISON-20260729.json",
    "CONTEXT-SHARD-POLICIES-20260729.json",
    "GRAPH-MODEL-SENSITIVITY-20260729.json",
    "GRAPH-REPETITIONS-QWEN25-20260729.json",
    "INCREMENTAL-MEMORY-LIFECYCLE-20260729.json",
    "INCREMENTAL-MEMORY-PERSISTENCE-20260729.json",
    "INGESTION-latest.json",
    "JCODE-COMMON-3X-20260729.json",
    "MEM0-COMMON-3X-20260729.json",
    "P3-DELETION-COMPACTION-20260729.json",
    "P3-DERIVED-INDEX-CRASH-MATRIX-20260729.json",
    "P3-HARNESS-QWEN25-14B-20260730.json",
    "P3-TEMPORAL-SCORING-QWEN25-14B-20260729.json",
    "PROTOTYPE-latest.json",
    "TOPOLOGY-latest.json",
)


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    missing = [name for name in SAFE_RESULTS if not (RESULTS / name).is_file()]
    if missing:
        print(f"missing reviewed result files: {', '.join(missing)}")
        return 1

    expected = set(SAFE_RESULTS)
    for stale in DESTINATION.glob("*.json"):
        if stale.name not in expected:
            stale.unlink()

    for name in SAFE_RESULTS:
        source = RESULTS / name
        destination = DESTINATION / name
        shutil.copyfile(source, destination)
        print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
