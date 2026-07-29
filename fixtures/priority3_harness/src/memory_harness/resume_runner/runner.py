from collections.abc import Callable, Iterable
from pathlib import Path

from .checkpoint import load_completed, save_completed
from .contracts import RunSummary


def run_items(
    item_ids: Iterable[str],
    processor: Callable[[str], None],
    checkpoint_path: Path,
) -> RunSummary:
    completed = load_completed(checkpoint_path)
    skipped = 0
    processed = 0
    for item_id in item_ids:
        if item_id in completed:
            skipped += 1
            continue
        processor(item_id)
        completed.add(item_id)
        processed += 1
    save_completed(checkpoint_path, completed)
    return RunSummary(completed=processed, skipped=skipped, failed=0)
