#!/usr/bin/env python3
"""Build a privacy-preserving manifest for local raw result artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "published" / "raw-evidence-manifest.json"
PUBLISHED_RESULTS = RESULTS / "published" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    artifacts = []
    for path in sorted(RESULTS.glob("*.json")):
        published_copy = PUBLISHED_RESULTS / path.name
        is_published = published_copy.is_file() and sha256(published_copy) == sha256(path)
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "published": is_published,
                **(
                    {"published_path": published_copy.relative_to(ROOT).as_posix()}
                    if is_published
                    else {
                        "reason": "raw dataset or provider output requires a separate license and privacy review"
                    }
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "artifact_count": len(artifacts),
        "total_bytes": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
