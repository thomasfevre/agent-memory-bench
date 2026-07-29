#!/usr/bin/env python3
"""Download and verify public benchmark datasets pinned in the upstream lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "upstreams.lock.json"
DEFAULT_DESTINATION = ROOT / ".cache" / "datasets"
BUFFER_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(BUFFER_SIZE):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, entry: dict[str, Any]) -> None:
    actual_size = path.stat().st_size
    if actual_size != int(entry["bytes"]):
        raise RuntimeError(
            f"size mismatch for {entry['name']}: "
            f"expected {entry['bytes']}, got {actual_size}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != entry["sha256"]:
        raise RuntimeError(
            f"sha256 mismatch for {entry['name']}: {actual_sha256}"
        )


def download(entry: dict[str, Any], destination_root: Path) -> None:
    destination = destination_root / entry["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify(destination, entry)
        print(f"{entry['name']}: verified {destination}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            entry["url"],
            headers={"User-Agent": "agent-memory-bench/1.0"},
        )
        with urllib.request.urlopen(request) as response, temporary.open(
            "wb"
        ) as handle:
            while block := response.read(BUFFER_SIZE):
                handle.write(block)
        verify(temporary, entry)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{entry['name']}: downloaded {destination}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    selected = {item.lower() for item in args.only}
    entries = [
        entry
        for entry in payload["datasets"]
        if not selected or entry["name"].lower() in selected
    ]
    if selected and len(entries) != len(selected):
        known = {entry["name"].lower() for entry in payload["datasets"]}
        missing = sorted(selected - known)
        raise ValueError(f"unknown datasets: {', '.join(missing)}")

    args.destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        download(entry, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
