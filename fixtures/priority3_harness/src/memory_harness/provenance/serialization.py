from typing import Any

from .contracts import Evidence


def to_payload(evidence: Evidence) -> dict[str, Any]:
    return {"text": evidence.text, "source_ids": list(evidence.source_ids)}


def from_payload(payload: str | dict[str, Any]) -> Evidence:
    if isinstance(payload, str):
        return Evidence(text=payload, source_ids=())
    return Evidence(
        text=str(payload["text"]),
        source_ids=tuple(payload.get("source_ids", [])),
    )
