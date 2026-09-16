from __future__ import annotations

from .contracts import RawSkillOutput, SkillContext, SkillVersion


class MinimumPositiveSleevesRegime:
    def __init__(self, version: SkillVersion) -> None:
        self.version = version

    def execute(self, context: SkillContext) -> RawSkillOutput:
        minimum = int(context.binding.config.get("minimum_positive_sleeves", 3))
        if not 1 <= minimum <= 100:
            raise ValueError("minimum_positive_sleeves must be in [1, 100]")
        signals = context.state.get("signals", {})
        positive = [symbol for symbol, signal in signals.items() if bool(signal.get("eligible"))]
        blocked = len(positive) < minimum
        reason = (
            f"only {len(positive)} eligible positive sleeves; {minimum} required"
            if blocked
            else ""
        )
        return RawSkillOutput(
            result_type="regime_state",
            payload={
                "regime_id": "TREND_ON" if not blocked else "DEFENSIVE_CASH",
                "positive_sleeves": positive,
                "minimum_positive_sleeves": minimum,
                "risk_scalar": 0.0 if blocked else 1.0,
                "blocked": blocked,
                "blocking_reasons": [reason] if reason else [],
            },
            confidence=min(1.0, len(positive) / minimum),
            reasons=(reason,) if reason else (),
        )
