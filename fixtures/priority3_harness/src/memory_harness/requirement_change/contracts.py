from dataclasses import dataclass


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str
