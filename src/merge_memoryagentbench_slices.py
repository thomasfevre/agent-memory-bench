from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graph_benchmark_common import write_result
from memoryagentbench_slice import summarize


def merge_slices(paths: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text()) for path in paths]
    rows = [
        row
        for payload in payloads
        for row in payload["rows"]
    ]
    unique_rows = {
        (int(row["question_index"]), row["strategy"]): row
        for row in rows
    }
    merged_rows = [
        unique_rows[key]
        for key in sorted(unique_rows)
    ]
    question_indices = sorted({int(row["question_index"]) for row in merged_rows})
    first_manifest = payloads[0]["manifest"]
    return {
        "manifest": {
            **first_manifest,
            "questions": len(question_indices),
            "question_indices": question_indices,
            "source_files": [path.name for path in paths],
            "note": "Merged bounded local slices, not a reproduction of the paper table.",
        },
        "rows": merged_rows,
        "summaries": summarize(merged_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_result(args.output, merge_slices(args.inputs))


if __name__ == "__main__":
    main()
