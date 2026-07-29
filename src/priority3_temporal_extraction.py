#!/usr/bin/env python3
"""Generate and evaluate Priority 3 temporal extraction observations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ADVERSARIAL_TYPES = (
    "correction",
    "late_arrival",
    "duplicate",
    "expiration",
    "retraction",
    "low_confidence_contradiction",
    "shard_approval",
    "shard_rejection",
)


def iso(day: datetime) -> str:
    return day.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def event_text(expected: dict[str, Any]) -> str:
    entity = expected["entity_key"]
    value = expected["value"]
    asserted = expected["asserted_at"]
    effective = expected["effective_from"]
    source = expected["source_id"]
    target = expected.get("target_event_id")
    event_type = expected["event_type"]
    if event_type == "assertion":
        return (
            f"Source {source} reported on {asserted} that {entity} became "
            f"{value}, effective from {effective}, with confidence "
            f"{expected['confidence']:.2f}."
        )
    if event_type == "correction":
        return (
            f"Correction from {source}, asserted {asserted}: replace event "
            f"{target} for {entity} with value {value}. The correction was "
            f"effective earlier, from {effective}, confidence "
            f"{expected['confidence']:.2f}."
        )
    if event_type == "late_arrival":
        return (
            f"Late-arriving source {source} was received on {asserted}. It "
            f"states that {entity} was {value} from {effective}; confidence "
            f"{expected['confidence']:.2f}."
        )
    if event_type == "duplicate":
        return (
            f"Retry from {source} on {asserted}: this is a duplicate of "
            f"{target}. It repeats {entity} = {value}, effective "
            f"{effective}, confidence {expected['confidence']:.2f}."
        )
    if event_type == "expiration":
        return (
            f"Expiration notice {source}, asserted {asserted}: event {target} "
            f"for {entity} stops being valid after "
            f"{expected['effective_until']}. Its recorded value was {value}."
        )
    if event_type == "retraction":
        return (
            f"Retraction {source}, asserted {asserted}: withdraw event "
            f"{target} about {entity} with effect from {effective}. The "
            f"withdrawn value was {value}."
        )
    if event_type == "low_confidence_contradiction":
        return (
            f"Unverified source {source} claimed on {asserted} that {entity} "
            f"was instead {value} from {effective}. This contradicts the "
            f"authoritative record and has confidence "
            f"{expected['confidence']:.2f}."
        )
    decision = "approved" if event_type == "shard_approval" else "rejected"
    return (
        f"Human review {source}, recorded {asserted}: context shard {target} "
        f"for {entity} is {decision} from {effective}. The reviewed shard "
        f"value is {value}, confidence {expected['confidence']:.2f}."
    )


def make_expected(
    *,
    entity_index: int,
    event_index: int,
    event_type: str,
    initial_id: str,
    asserted_at: datetime,
    effective_from: datetime,
) -> dict[str, Any]:
    entity_key = f"entity-{entity_index:02d}.operating_mode"
    initial_value = f"mode-{entity_index:02d}-v1"
    value = initial_value
    target_event_id = None
    confidence = 0.96
    effective_until = None
    if event_type == "correction":
        value = f"mode-{entity_index:02d}-v2"
        target_event_id = initial_id
    elif event_type == "late_arrival":
        value = f"mode-{entity_index:02d}-late"
    elif event_type == "duplicate":
        target_event_id = initial_id
    elif event_type == "expiration":
        target_event_id = initial_id
        effective_until = iso(effective_from + timedelta(days=4))
    elif event_type == "retraction":
        target_event_id = initial_id
    elif event_type == "low_confidence_contradiction":
        value = f"mode-{entity_index:02d}-rumor"
        confidence = 0.55
    elif event_type in {"shard_approval", "shard_rejection"}:
        value = f"team-rule-{entity_index:02d}"
        target_event_id = f"shard-{entity_index:02d}"
        confidence = 1.0
    return {
        "entity_key": entity_key,
        "value": value,
        "asserted_at": iso(asserted_at),
        "effective_from": iso(effective_from),
        "effective_until": effective_until,
        "event_type": event_type,
        "target_event_id": target_event_id,
        "confidence": confidence,
        "source_id": f"p3-source-{entity_index:02d}-{event_index}",
    }


def generate_observations() -> list[dict[str, Any]]:
    base = datetime(2026, 1, 1, 9, tzinfo=timezone.utc)
    observations = []
    sequence = 1
    for entity_index in range(1, 21):
        initial_id = f"p3-event-{sequence:03d}"
        initial_day = base + timedelta(days=entity_index)
        initial = make_expected(
            entity_index=entity_index,
            event_index=1,
            event_type="assertion",
            initial_id=initial_id,
            asserted_at=initial_day,
            effective_from=initial_day,
        )
        observations.append(
            {
                "id": initial_id,
                "canonical_sequence": sequence,
                "text": event_text(initial),
                "expected": initial,
            }
        )
        sequence += 1

        for event_index, offset in ((2, 0), (3, 3)):
            event_type = ADVERSARIAL_TYPES[
                (entity_index - 1 + offset) % len(ADVERSARIAL_TYPES)
            ]
            asserted_at = base + timedelta(
                days=45 + entity_index * 2 + event_index
            )
            effective_from = asserted_at
            if event_type in {"correction", "late_arrival"}:
                effective_from -= timedelta(days=20 + entity_index % 5)
            expected = make_expected(
                entity_index=entity_index,
                event_index=event_index,
                event_type=event_type,
                initial_id=initial_id,
                asserted_at=asserted_at,
                effective_from=effective_from,
            )
            event_id = f"p3-event-{sequence:03d}"
            observations.append(
                {
                    "id": event_id,
                    "canonical_sequence": sequence,
                    "text": event_text(expected),
                    "expected": expected,
                }
            )
            sequence += 1
    return observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in generate_observations()
        ),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
