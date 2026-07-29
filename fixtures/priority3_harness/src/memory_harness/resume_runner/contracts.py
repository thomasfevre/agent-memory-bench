from dataclasses import dataclass


@dataclass(frozen=True)
class RunSummary:
    completed: int
    skipped: int
    failed: int
