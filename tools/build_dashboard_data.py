#!/usr/bin/env python3
"""Copy the canonical public registry into the static dashboard.

The site intentionally consumes a generated copy. CI can therefore prove that
the dashboard corresponds exactly to the versioned benchmark registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "published" / "registry.json"
DESTINATION = ROOT / "site" / "data" / "registry.json"
MANIFEST_SOURCE = ROOT / "results" / "published" / "raw-evidence-manifest.json"
MANIFEST_DESTINATION = ROOT / "site" / "data" / "raw-evidence-manifest.json"


def canonical_bytes(path: Path) -> bytes:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode(
        "utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated dashboard copy is missing or stale",
    )
    args = parser.parse_args()

    pairs = [
        (SOURCE, DESTINATION),
        (MANIFEST_SOURCE, MANIFEST_DESTINATION),
    ]
    if args.check:
        stale = [
            destination.relative_to(ROOT).as_posix()
            for source, destination in pairs
            if not destination.exists() or destination.read_bytes() != canonical_bytes(source)
        ]
        if stale:
            print(
                f"{', '.join(stale)} is stale; run tools/build_dashboard_data.py"
            )
            return 1
        print("dashboard registry is current")
        return 0

    for source, destination in pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_bytes(source))
        print(destination.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
