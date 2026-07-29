from .contracts import AccessDecision


def render_decision(decision: AccessDecision) -> dict[str, str | bool]:
    return {"allowed": decision.allowed, "reason": decision.reason}
