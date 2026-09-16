"""Legacy trial heuristics, explicitly not formal PBO or deflated Sharpe tests.

PBO proxy is the fraction of trials with nonpositive reported returns. DSR
proxy is max(return) minus the upper median reported return. Neither estimates
its namesake statistic. Mandate pass fraction does not establish walk-forward
validation, independence, or out-of-sample provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agipot_stock_prediction.research.contracts import finite_number


@dataclass(frozen=True)
class OverfitGuardResult:
    passed: bool
    blockers: tuple[str, ...]
    walk_forward_pass_rate: float
    pbo_proxy: float
    dsr_proxy: float
    test_performance_not_collapsed: bool
    all_trials_recorded: bool
    validation_level: str = "heuristic_only"

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "agipot_overfit_heuristics_v1",
            "passed": self.passed,
            "blockers": list(self.blockers),
            "walk_forward_pass_rate": self.walk_forward_pass_rate,
            "pbo_proxy": self.pbo_proxy,
            "dsr_proxy": self.dsr_proxy,
            "test_performance_not_collapsed": self.test_performance_not_collapsed,
            "all_trials_recorded": self.all_trials_recorded,
            "validation_level": self.validation_level,
            "formal_pbo_computed": False,
            "formal_dsr_computed": False,
            "out_of_sample_verified": False,
            "limitations": [
                "walk_forward_pass_rate is the supplied trial mandate-pass fraction",
                "pbo_proxy is the nonpositive-return fraction, not formal PBO",
                "dsr_proxy is max minus upper-median return, not deflated Sharpe",
                "input returns, trial completeness, and out-of-sample provenance are caller assertions",
            ],
        }


def evaluate_overfit_guard(
    trials: Sequence[Mapping[str, Any]],
    *,
    all_trials_recorded: bool,
    configured_max_pbo: float = 0.25,
    configured_min_dsr: float = 0.0,
) -> OverfitGuardResult:
    """Screen supplied trials; ``passed`` means only these heuristic rules pass."""
    max_pbo = finite_number(configured_max_pbo, "overfit.configured_max_pbo")
    min_dsr = finite_number(configured_min_dsr, "overfit.configured_min_dsr")
    if not 0 <= max_pbo <= 1 or min_dsr < 0:
        raise ValueError("heuristic thresholds require max_pbo in [0, 1] and min_dsr >= 0")
    if type(all_trials_recorded) is not bool:
        raise ValueError("all_trials_recorded must be a boolean")
    returns: list[float] = []
    symbols: set[str] = set()
    years: set[int] = set()
    for index, trial in enumerate(trials):
        if type(trial.get("mandate_passed")) is not bool:
            raise ValueError(f"trials[{index}].mandate_passed must be present and boolean")
        if "geometric_daily_return" not in trial:
            raise ValueError(f"trials[{index}].geometric_daily_return is required")
        returns.append(finite_number(trial["geometric_daily_return"], f"trials[{index}].geometric_daily_return"))
        trial_symbols = trial.get("symbols")
        if not isinstance(trial_symbols, (list, tuple)) or not trial_symbols or any(not isinstance(item, str) or not item.strip() for item in trial_symbols):
            raise ValueError(f"trials[{index}].symbols must be a nonempty symbol list")
        rows = trial.get("yearly_results")
        if not isinstance(rows, (list, tuple)) or not rows or any(not isinstance(row, Mapping) or type(row.get("year")) is not int for row in rows):
            raise ValueError(f"trials[{index}].yearly_results must contain year evidence")
        symbols.update(trial_symbols)
        years.update(row["year"] for row in rows)
    total = len(trials)
    passed_count = sum(trial["mandate_passed"] for trial in trials)
    walk_forward_pass_rate = passed_count / total if total else 0.0
    pbo_proxy = 1.0 - sum(value > 0 for value in returns) / total if total else 1.0
    dsr_proxy = max(returns) - sorted(returns)[len(returns) // 2] if returns else -1.0
    test_performance_not_collapsed = bool(returns and max(returns) > -0.05)
    blockers: list[str] = []
    if not total:
        blockers.append("no_trials")
    if walk_forward_pass_rate < 0.70:
        blockers.append("trial_mandate_pass_rate_below_min")
    if pbo_proxy > max_pbo:
        blockers.append("pbo_proxy_above_max")
    if dsr_proxy < min_dsr:
        blockers.append("dsr_proxy_below_min")
    if not test_performance_not_collapsed:
        blockers.append("reported_performance_collapsed")
    if not all_trials_recorded:
        blockers.append("all_trials_not_recorded")
    if total >= 100 and passed_count <= 1:
        blockers.append("backtest_overfitting_risk_high")
    if total and len(symbols) <= 1 and len(years) <= 1:
        blockers.append("single_slice_overfit_risk")
    return OverfitGuardResult(
        passed=not blockers,
        blockers=tuple(blockers),
        walk_forward_pass_rate=walk_forward_pass_rate,
        pbo_proxy=pbo_proxy,
        dsr_proxy=dsr_proxy,
        test_performance_not_collapsed=test_performance_not_collapsed,
        all_trials_recorded=all_trials_recorded,
    )


__all__ = ["OverfitGuardResult", "evaluate_overfit_guard"]
