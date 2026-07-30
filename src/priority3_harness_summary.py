from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TOKEN_CEILINGS = (25_000, 50_000, 100_000)
TIME_CEILINGS_SECONDS = (300, 600, 1200)
TOOL_CEILINGS = (20, 40, 80)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        grouped[str(row["attempt"]["harness"])].append(row)

    harnesses = []
    for harness, rows in sorted(grouped.items()):
        distinct_tasks = {str(row["task_id"]) for row in rows}
        tasks_completed = sum(bool(row["task_complete"]) for row in rows)
        known_tokens = [
            int(row["attempt"]["total_tokens"])
            for row in rows
            if row["attempt"].get("total_tokens") is not None
        ]
        production_files_changed = sum(
            len(row["production_files_changed"]) for row in rows
        )
        no_change_failures = (
            len(distinct_tasks) >= 2
            and tasks_completed == 0
            and production_files_changed == 0
        )
        harnesses.append(
            {
                "harness": harness,
                "classification": (
                    "tool_protocol_incompatible"
                    if no_change_failures
                    else "completed"
                    if tasks_completed == len(rows)
                    else "operational_failure"
                ),
                "attempts": len(rows),
                "distinct_tasks": len(distinct_tasks),
                "tasks_completed": tasks_completed,
                "public_tests_passed": sum(
                    int(row["public"]["passed"]) for row in rows
                ),
                "public_tests_total": sum(
                    int(row["public"]["tests"]) for row in rows
                ),
                "hidden_tests_passed": sum(
                    int(row["hidden"]["passed"]) for row in rows
                ),
                "hidden_tests_total": sum(
                    int(row["hidden"]["tests"]) for row in rows
                ),
                "production_files_changed": production_files_changed,
                "known_total_tokens": (
                    sum(known_tokens) if known_tokens else None
                ),
                "known_wall_time_seconds": round(
                    sum(
                        float(row["attempt"]["wall_time_seconds"])
                        for row in rows
                    ),
                    3,
                ),
            }
        )

    budget_rows = []
    for token_ceiling in TOKEN_CEILINGS:
        eligible = [
            row
            for row in attempts
            if row["attempt"].get("total_tokens") is not None
            and int(row["attempt"]["total_tokens"]) <= token_ceiling
        ]
        budget_rows.append(
            {
                "token_ceiling": token_ceiling,
                "eligible_attempts": len(eligible),
                "tasks_completed": sum(
                    bool(row["task_complete"]) for row in eligible
                ),
                "correct_tasks_per_100k_tokens": 0.0,
            }
        )

    time_rows = []
    for ceiling in TIME_CEILINGS_SECONDS:
        eligible = [
            row
            for row in attempts
            if float(row["attempt"]["wall_time_seconds"]) <= ceiling
        ]
        time_rows.append(
            {
                "wall_time_ceiling_seconds": ceiling,
                "eligible_attempts": len(eligible),
                "tasks_completed": sum(
                    bool(row["task_complete"]) for row in eligible
                ),
                "correct_tasks_per_hour": 0.0,
            }
        )

    return {
        "harnesses": harnesses,
        "fixed_budget": {
            "rows": budget_rows,
            "time_rows": time_rows,
            "tool_call_ceilings": list(TOOL_CEILINGS),
            "tool_phase_proportions": None,
            "pareto_efficient_configurations": [],
            "note": (
                "No attempt completed a task, so quality-per-cost is zero. "
                "Codex token totals and reliable tool-call phase attribution "
                "were unavailable after interrupted unsupported-call loops."
            ),
        },
    }


def compact_attempt(payload: dict[str, Any]) -> dict[str, Any]:
    attempt = payload["attempt"]
    return {
        "task_id": payload["task_id"],
        "harness": attempt["harness"],
        "harness_version": attempt["harness_version"],
        "repetition": attempt["repetition"],
        "wall_time_seconds": attempt["wall_time_seconds"],
        "input_tokens": attempt["input_tokens"],
        "output_tokens": attempt["output_tokens"],
        "total_tokens": attempt["total_tokens"],
        "tool_calls": attempt["tool_calls"],
        "exit_code": attempt["exit_code"],
        "changed_files": payload["changed_files"],
        "production_files_changed": payload["production_files_changed"],
        "public_tests_passed": payload["public"]["passed"],
        "public_tests_total": payload["public"]["tests"],
        "hidden_tests_passed": payload["hidden"]["passed"],
        "hidden_tests_total": payload["hidden"]["tests"],
        "task_complete": payload["task_complete"],
        "observation": attempt["observation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--claude-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.results_dir.glob("*.json")
        if path.name != args.claude_result.name
    )
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
        if "attempt" in json.loads(path.read_text(encoding="utf-8"))
    ]
    summary = summarize_attempts(payloads)
    claude = json.loads(args.claude_result.read_text(encoding="utf-8"))
    summary["harnesses"].append(
        {
            "harness": claude["harness"],
            "classification": claude["classification"],
            "attempts": len(claude["attempts"]),
            "distinct_tasks": 0,
            "tasks_completed": 0,
            "public_tests_passed": 0,
            "public_tests_total": 0,
            "hidden_tests_passed": 0,
            "hidden_tests_total": 0,
            "production_files_changed": 0,
            "known_total_tokens": None,
            "known_wall_time_seconds": round(
                sum(row["wall_time_seconds"] for row in claude["attempts"]),
                3,
            ),
        }
    )
    summary["harnesses"].sort(key=lambda row: row["harness"])
    result = {
        "schema_version": 1,
        "campaign": "priority3-coding-harness-qwen25-14b",
        "date": "2026-07-30",
        "fixture_commit": "cf0dfa68b028f8dab5b39d0a5dddd9e14f2298ea",
        "model": "qwen2.5:14b",
        "network_policy": "localhost-only",
        "attempts": [compact_attempt(payload) for payload in payloads],
        "harnesses": summary["harnesses"],
        "fixed_budget": summary["fixed_budget"],
        "claude_provider_incompatibility": claude,
        "raw_manifest": [
            {"path": path.name, "sha256": sha256(path)} for path in paths
        ]
        + [
            {
                "path": args.claude_result.name,
                "sha256": sha256(args.claude_result),
            }
        ],
        "conclusion": (
            "The pinned local model could answer in prose but did not complete "
            "repository work through any harness. Two independent tasks "
            "reproduced no-change tool-protocol failures for jcode, Codex and "
            "Letta; Claude Code failed two provider handshakes because the "
            "adapter requested unsupported thinking."
        ),
        "limits": [
            "This is a compatibility result for qwen2.5:14b, not a quality ranking of the harnesses with their recommended models.",
            "Codex token totals are unavailable for interrupted local-provider runs.",
            "Only two tasks were executed after reproducible no-change failures made the remaining task matrix non-informative.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
