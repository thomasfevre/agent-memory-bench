from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    text: str
    source_ids: tuple[str, ...]
