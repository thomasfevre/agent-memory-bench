from dataclasses import dataclass


@dataclass(frozen=True)
class CampaignResult:
    applied: int
    replayed: int
