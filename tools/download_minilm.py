#!/usr/bin/env python3
"""Download the pinned MiniLM ONNX artifacts used by the common benchmark."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / ".cache" / "models" / "all-MiniLM-L6-v2"
BASE_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/"
    "resolve/main"
)
ARTIFACTS = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(os.environ.get("AMB_MINILM_DIR", DEFAULT_DESTINATION)),
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    for remote_name, expected_hash in ARTIFACTS.items():
        destination = args.destination / Path(remote_name).name
        if destination.is_file() and sha256(destination) == expected_hash:
            print(f"verified {destination}")
            continue
        with tempfile.NamedTemporaryFile(
            dir=args.destination, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{remote_name}", temporary_path)
            actual_hash = sha256(temporary_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"checksum mismatch for {remote_name}: {actual_hash}"
                )
            temporary_path.replace(destination)
            print(f"downloaded {destination}")
        finally:
            temporary_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
