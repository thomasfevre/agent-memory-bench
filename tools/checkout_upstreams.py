#!/usr/bin/env python3
"""Materialize pinned public upstream repositories in the ignored cache."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "upstreams.lock.json"
DEFAULT_DESTINATION = ROOT / ".cache" / "upstreams"


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def materialize(entry: dict[str, Any], destination_root: Path) -> None:
    destination = destination_root / entry["name"]
    created = False
    if not destination.exists():
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                entry["repository"],
                str(destination),
            ]
        )
        created = True
    if not created and run(["git", "status", "--porcelain"], destination):
        raise RuntimeError(f"refusing to change dirty checkout: {destination}")
    run(["git", "fetch", "origin", entry["commit"], "--depth=1"], destination)
    run(["git", "checkout", "--detach", entry["commit"]], destination)
    actual = run(["git", "rev-parse", "HEAD"], destination)
    if actual != entry["commit"]:
        raise RuntimeError(f"checkout mismatch for {entry['name']}: {actual}")
    print(f"{entry['name']}: {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    selected = {item.lower() for item in args.only}
    entries = [
        entry
        for entry in payload["repositories"]
        if not selected or entry["name"].lower() in selected
    ]
    if selected and len(entries) != len(selected):
        known = {entry["name"].lower() for entry in payload["repositories"]}
        missing = sorted(selected - known)
        raise ValueError(f"unknown upstreams: {', '.join(missing)}")

    args.destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        materialize(entry, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
