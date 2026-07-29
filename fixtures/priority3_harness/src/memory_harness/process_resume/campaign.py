from collections.abc import Callable, Iterable
from pathlib import Path

from .contracts import CampaignResult
from .journal import append_ack, load_acks


def apply_campaign(
    event_ids: Iterable[str],
    apply_event: Callable[[str], None],
    journal_path: Path,
) -> CampaignResult:
    acknowledged = {row["event_id"] for row in load_acks(journal_path)}
    applied = 0
    replayed = 0
    for event_id in event_ids:
        if event_id in acknowledged:
            replayed += 1
            continue
        apply_event(event_id)
        append_ack(journal_path, {"event_id": event_id})
        applied += 1
    return CampaignResult(applied=applied, replayed=replayed)
