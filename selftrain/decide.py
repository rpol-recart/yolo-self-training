"""Round acceptance and stopping rules (pure functions)."""

from __future__ import annotations


def accept_round(candidate: dict, best: dict, acfg: dict) -> tuple[bool, list[str]]:
    """candidate / best: {'overall': {metric: v}, 'groups': {g: {metric: v}}}.

    Accept iff overall metric improves by >= min_delta AND no group shared by both
    drops by more than group_tolerance.
    """
    m = acfg["metric"]
    reasons = []
    cand_v, best_v = candidate["overall"][m], best["overall"][m]
    ok = cand_v >= best_v + acfg["min_delta"]
    if not ok:
        reasons.append(f"{m} {cand_v:.4f} < best {best_v:.4f} + min_delta {acfg['min_delta']}")
    for g, bv in best.get("groups", {}).items():
        cv = candidate.get("groups", {}).get(g)
        if cv is None:
            continue
        drop = bv[m] - cv[m]
        if drop > acfg["group_tolerance"]:
            ok = False
            reasons.append(f"group {g}: {m} dropped {drop:.4f} > tolerance {acfg['group_tolerance']}")
    if ok:
        reasons.append(f"{m} {best_v:.4f} -> {cand_v:.4f}")
    return ok, reasons


def should_stop(history: list[dict], lcfg: dict) -> tuple[bool, str]:
    """history: rounds >= 1 in order, each with an 'accepted' flag (round 0 excluded)."""
    if len(history) >= lcfg["max_rounds"]:
        return True, f"max_rounds={lcfg['max_rounds']} reached"
    streak = 0
    for h in reversed(history):
        if h["accepted"]:
            break
        streak += 1
    if streak >= lcfg["patience"]:
        return True, f"{streak} rounds in a row without improvement (patience={lcfg['patience']})"
    return False, ""
