#!/usr/bin/env python3
"""Compare Context Shard promotion and review policies on the common corpus."""

from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmark import load_jsonl
from graph_benchmark_common import write_result


STATIC_POLICIES = (
    "auto_promote_repeated",
    "approved_only",
    "review_registry",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-occurrences", type=int, default=2)
    parser.add_argument("--recheck-days", type=int, default=30)
    return parser.parse_args()


def policy_decision(
    shard: dict[str, Any],
    policy: str,
    minimum_occurrences: int,
) -> bool | None:
    if policy == "auto_promote_repeated":
        return bool(shard["occurrences"] >= minimum_occurrences)
    if policy == "approved_only":
        return True if shard["review"] == "approved" else None
    if policy == "review_registry":
        if shard["review"] == "approved":
            return True
        if shard["review"] == "rejected":
            return False
        return None
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_static_policy(
    shards: list[dict[str, Any]],
    policy: str,
    minimum_occurrences: int,
) -> dict[str, Any]:
    repeated = [
        shard
        for shard in shards
        if shard["occurrences"] >= minimum_occurrences
    ]
    rows = []
    for shard in repeated:
        prediction = policy_decision(shard, policy, minimum_occurrences)
        expected = shard["review"] == "approved"
        rows.append(
            {
                "shard_id": shard["id"],
                "expected_approved": expected,
                "predicted_approved": prediction,
                "known": prediction is not None,
                "correct": prediction == expected,
            }
        )
    known = [row for row in rows if row["known"]]
    active = [
        shard
        for shard in repeated
        if policy_decision(shard, policy, minimum_occurrences) is True
    ]
    rejected_count = sum(
        shard["review"] == "rejected" for shard in repeated
    )
    return {
        "policy": policy,
        "candidate_shards": len(repeated),
        "decision_coverage": len(known) / len(rows),
        "strict_decision_accuracy": sum(row["correct"] for row in rows)
        / len(rows),
        "known_decision_accuracy": (
            sum(row["correct"] for row in known) / len(known)
            if known
            else 0.0
        ),
        "active_shards": len(active),
        "active_precision": (
            sum(shard["review"] == "approved" for shard in active) / len(active)
            if active
            else 0.0
        ),
        "rejected_activation_rate": (
            sum(shard["review"] == "rejected" for shard in active)
            / rejected_count
            if rejected_count
            else 0.0
        ),
        "rows": rows,
    }


def run_lifecycle(
    events: list[dict[str, Any]],
    minimum_occurrences: int,
    recheck_days: int,
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    trace = []
    requeues = 0
    unsafe_activations = 0

    for event in sorted(events, key=lambda item: item["timestamp"]):
        shard_id = event["shard_id"]
        state = states.setdefault(
            shard_id,
            {
                "occurrences": 0,
                "status": "observed",
                "active": False,
                "review_due_at": None,
            },
        )
        timestamp = date.fromisoformat(event["timestamp"])

        if event["event"] == "observe":
            state["occurrences"] += 1
            due = state["review_due_at"]
            if (
                state["status"] == "rejected"
                and due is not None
                and timestamp >= date.fromisoformat(due)
            ):
                state["status"] = "pending_review"
                state["review_due_at"] = None
                requeues += 1
            elif (
                state["occurrences"] >= minimum_occurrences
                and state["status"] in {"observed", "deferred"}
            ):
                state["status"] = "pending_review"
            state["active"] = state["status"] == "approved"
        elif event["event"] == "review":
            decision = event["decision"]
            state["status"] = decision
            state["active"] = decision == "approved"
            state["review_due_at"] = (
                (timestamp + timedelta(days=recheck_days)).isoformat()
                if decision == "rejected"
                else None
            )
        else:
            raise ValueError(f"Unknown lifecycle event: {event['event']}")

        if state["active"] and state["status"] == "rejected":
            unsafe_activations += 1
        trace.append(
            {
                "timestamp": event["timestamp"],
                "shard_id": shard_id,
                "event": event["event"],
                "synthetic": event["synthetic"],
                **state,
            }
        )

    return {
        "requeues_after_cooldown": requeues,
        "unsafe_activation_events": unsafe_activations,
        "final_active_shards": sorted(
            shard_id for shard_id, state in states.items() if state["active"]
        ),
        "final_states": states,
        "trace": trace,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    shards = load_jsonl(args.shards)
    events = load_jsonl(args.events)
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Context Shard policy mechanics on the common corpus",
            "shards_file": str(args.shards),
            "shards_sha256": hashlib.sha256(args.shards.read_bytes()).hexdigest(),
            "events_file": str(args.events),
            "events_sha256": hashlib.sha256(args.events.read_bytes()).hexdigest(),
            "minimum_occurrences": args.minimum_occurrences,
            "recheck_days": args.recheck_days,
            "static_policies": list(STATIC_POLICIES),
            "synthetic_scope": (
                "Review timestamps and the post-rejection recurrence are synthetic "
                "policy events. Original observations retain synthetic=false."
            ),
            "interpretation_limit": (
                "This validates state transitions and promotion safety on three "
                "gold shards. It does not measure real reviewer effort or long-term "
                "organizational usefulness."
            ),
        },
        "static_policies": [
            evaluate_static_policy(
                shards,
                policy,
                args.minimum_occurrences,
            )
            for policy in STATIC_POLICIES
        ],
        "lifecycle": run_lifecycle(
            events,
            args.minimum_occurrences,
            args.recheck_days,
        ),
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
