from __future__ import annotations

from enum import Enum


class AutonomyMode(str, Enum):
    DISABLED = "DISABLED"
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    LIMITED_AUTONOMY = "LIMITED_AUTONOMY"
    REDUCE_ONLY = "REDUCE_ONLY"
    LOCKDOWN = "LOCKDOWN"


PROMOTION_ORDER = (
    AutonomyMode.RESEARCH,
    AutonomyMode.SHADOW,
    AutonomyMode.PAPER,
    AutonomyMode.HUMAN_APPROVAL,
    AutonomyMode.LIMITED_AUTONOMY,
)


def coerce_autonomy_mode(mode: AutonomyMode | str) -> AutonomyMode:
    if isinstance(mode, AutonomyMode):
        return mode
    return AutonomyMode(str(mode))


def next_promotion_mode(mode: AutonomyMode | str) -> AutonomyMode | None:
    current = coerce_autonomy_mode(mode)
    if current not in PROMOTION_ORDER:
        return None
    index = PROMOTION_ORDER.index(current)
    if index + 1 >= len(PROMOTION_ORDER):
        return None
    return PROMOTION_ORDER[index + 1]


__all__ = ["AutonomyMode", "PROMOTION_ORDER", "coerce_autonomy_mode", "next_promotion_mode"]
