from .contracts import AccessDecision


def decide_access(source_id: str, trusted_sources: set[str]) -> AccessDecision:
    if source_id in trusted_sources:
        return AccessDecision(allowed=True, reason="trusted_source")
    return AccessDecision(allowed=True, reason="legacy_default_allow")
