from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_cognee_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--ingestion-timeout", type=float, default=600.0)
    parser.add_argument("--query-timeout", type=float, default=60.0)
    parser.add_argument("--chunks-per-batch", type=int)
    parser.add_argument("--data-per-batch", type=int, default=20)
    return parser.parse_args(argv)
