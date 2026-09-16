"""Regression tests for research gates, numeric data, and allocation invariants."""

import math

import pytest

from agipot_stock_prediction.formulas.cost import evaluate_cost_gate
from agipot_stock_prediction.formulas.data_integrity import evaluate_data_integrity
from agipot_stock_prediction.formulas.ensemble import evaluate_ensemble
from agipot_stock_prediction.formulas.feedback import evaluate_feedback
from agipot_stock_prediction.formulas.intrinsic_value import evaluate_intrinsic_value
from agipot_stock_prediction.formulas.liquidity import evaluate_liquidity
from agipot_stock_prediction.formulas.moat import evaluate_moat
from agipot_stock_prediction.formulas.owner_earnings import evaluate_owner_earnings
from agipot_stock_prediction.formulas.promotion_formula import evaluate_promotion_formula, evaluate_trade_permission
from agipot_stock_prediction.formulas.quality import evaluate_quality
from agipot_stock_prediction.formulas.sizing import evaluate_sizing
from agipot_stock_prediction.formulas.universe import evaluate_universe
from agipot_stock_prediction.research.cost_model import apply_cost_to_return


@pytest.mark.parametrize("temperature", [0.25, 0.0001, 1e-300])
def test_ensemble_cap_survives_redistribution(temperature):
    result = evaluate_ensemble(
        skill_scores={"high": 1, "medium": 0.5, "low": 0},
        skill_reliabilities={"high": 1, "medium": 0.1, "low": 0},
        temperature=temperature,
        max_skill_weight=0.4,
    )
    weights = result.evidence["weights"]
    assert sum(weights.values()) == pytest.approx(1)
    assert max(weights.values()) <= 0.4 + 1e-14
    assert min(weights.values()) >= 0
    assert weights["high"] >= weights["medium"] >= weights["low"]
    assert result.score == pytest.approx(weights["high"] + 0.5 * weights["medium"])


def test_ensemble_tight_feasible_cap_is_uniform():
    result = evaluate_ensemble(skill_scores={str(i): 1 for i in range(4)},
                               skill_reliabilities={str(i): i / 4 for i in range(4)},
                               max_skill_weight=0.25)
    assert list(result.evidence["weights"].values()) == pytest.approx([0.25] * 4)


@pytest.mark.parametrize("kwargs", [{"max_skill_weight": 0.3}, {"max_skill_weight": 0},
                                    {"temperature": 0}, {"temperature": -1}])
def test_impossible_ensemble_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate_ensemble(skill_scores={"a": 1, "b": 1, "c": 1},
                          skill_reliabilities={"a": 1, "b": 0, "c": 0}, **kwargs)


def test_missing_skill_reliability_cannot_pass_with_high_scores():
    result = evaluate_ensemble(skill_scores={"a": 1, "b": 1, "c": 1}, skill_reliabilities={})
    assert not result.pass_gate
    assert "missing_skill_reliabilities" in result.blockers


@pytest.fixture
def integrity_inputs():
    return dict(point_in_time=True, quote_age_seconds=1, max_quote_age_seconds=30,
                prices=[100], quantities=[1], account_id_match=True, agentic_account=True)


def test_complete_integrity_evidence_passes(integrity_inputs):
    assert evaluate_data_integrity(**integrity_inputs).pass_gate


@pytest.mark.parametrize("change", [{"prices": []}, {"quantities": []}, {"prices": [], "quantities": []},
                                   {"quote_age_seconds": -1}, {"max_quote_age_seconds": -1},
                                   {"quantities": [1, 2]}, {"prices": [0]}])
def test_missing_or_invalid_integrity_evidence_blocks(integrity_inputs, change):
    assert not evaluate_data_integrity(**(integrity_inputs | change)).pass_gate


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, True])
def test_nonfinite_price_is_rejected(integrity_inputs, bad):
    with pytest.raises(ValueError):
        evaluate_data_integrity(**(integrity_inputs | {"prices": [bad]}))


def test_truthy_string_is_not_valid_boolean_evidence(integrity_inputs):
    with pytest.raises(ValueError, match="boolean"):
        evaluate_data_integrity(**(integrity_inputs | {"point_in_time": "false"}))


@pytest.fixture
def quality_inputs():
    return dict(roic=0.3, gross_margin=0.6, operating_margin=0.3,
                owner_earnings_growth_history=[0.1] * 5, owner_earnings_history=[100] * 5,
                net_debt_to_ebitda=0, interest_coverage=20, current_ratio=3, fcf=100,
                net_income=100, roic_minus_wacc=0.2, buyback_yield_when_undervalued=0.05,
                shares_outstanding_current=100, shares_outstanding_3y_ago=100)


def test_quality_needs_history_even_with_strong_current_metrics(quality_inputs):
    assert evaluate_quality(**quality_inputs).pass_gate
    result = evaluate_quality(**(quality_inputs | {"owner_earnings_history": []}))
    assert not result.pass_gate
    assert "missing_owner_earnings_history" in result.blockers


def test_moat_proxy_cannot_bypass_missing_fundamentals():
    result = evaluate_moat(gross_margin_history=[], roic_history=[], revenue_history=[],
                           fundamental_coverage=0.1, proxy_mode_score=1, min_moat_score=0.1)
    assert not result.pass_gate
    assert "insufficient_fundamental_coverage" in result.blockers
    assert "price_proxy_used" in result.warnings


def test_liquidity_requires_volume_even_when_thresholds_are_zero():
    result = evaluate_liquidity(close=100, volume_history=[], bid=100, ask=100,
                                min_median_adv=0, min_p20_adv=0)
    assert not result.pass_gate
    assert "missing_liquidity_volume" in result.blockers


def test_nan_threshold_cannot_silently_pass_universe_gate():
    with pytest.raises(ValueError, match="finite"):
        evaluate_universe(asset_type="stock", allowed_asset_types=["stock"], sector="tech",
                          circle_of_competence=["tech"], business_model_known=True,
                          data_coverage=1, min_data_coverage=math.nan, close=100,
                          volume_history=[100000] * 20, bid=99.99, ask=100.01)


def test_owner_earnings_preserves_maintenance_and_history_formula():
    result = evaluate_owner_earnings(net_income=100, depreciation=10, amortization=2,
                                     other_non_cash_charges=3, total_capex=20, maintenance_ratio=0.5,
                                     normalized_working_capital_need=5, stock_compensation_adjustment=4,
                                     owner_earnings_history=[80, 90, 95, 100, 110])
    assert result.pass_gate
    assert result.evidence["owner_earnings"] == pytest.approx(96)
    assert result.evidence["normalized_owner_earnings"] == pytest.approx(97)


def test_constant_earnings_intrinsic_value_matches_perpetuity():
    result = evaluate_intrinsic_value(price=50, normalized_owner_earnings_per_share=10,
                                     risk_free_rate=0.04, equity_risk_premium=0.06,
                                     historical_oe_growth=0, analyst_growth=0,
                                     reinvestment_rate=0, roic=0.1, terminal_growth=0)
    assert result.pass_gate
    assert result.evidence["intrinsic_value_low"] == pytest.approx(100)


def test_cost_gate_subtracts_commission_and_spread():
    result = evaluate_cost_gate(expected_return=0.02, order_notional=1000, spread_bps=20,
                               median_daily_dollar_volume=1_000_000, commission=2, impact_coefficient=0)
    assert result.pass_gate
    assert result.evidence["cost_return"] == pytest.approx(0.003)
    assert result.evidence["net_expected_return"] == pytest.approx(0.017)
    with pytest.raises(ValueError, match="negative"):
        apply_cost_to_return(gross_return=0.1, cost_bps=-10)


@pytest.fixture
def sizing_inputs():
    return dict(alpha=1, reliability_weighted_confidence=1, data_confidence=1,
                liquidity_multiplier=1, regime_multiplier=1, margin_of_safety=0.3,
                target_margin_of_safety=0.3, quality_score=1, permanent_loss_risk=0,
                expected_excess_return=0.1, variance=0.2, max_symbol_weight=0.1,
                liquidity_capacity_weight=0.05)


def test_sizing_honors_capacity_and_kelly(sizing_inputs):
    result = evaluate_sizing(**sizing_inputs)
    assert result.pass_gate
    assert result.evidence["kelly_weight"] == pytest.approx(0.075)
    assert result.evidence["target_weight"] == pytest.approx(0.05)


@pytest.mark.parametrize("change", [{"variance": 0}, {"variance": -1}, {"variance": math.nan},
                                   {"liquidity_capacity_weight": math.nan}, {"max_symbol_weight": -0.1}])
def test_sizing_rejects_invalid_risk_inputs(sizing_inputs, change):
    with pytest.raises(ValueError):
        evaluate_sizing(**(sizing_inputs | change))


def test_feedback_needs_paired_observations():
    result = evaluate_feedback(current_reliability=0.8, signal_values=[1] * 10,
                               future_net_returns=[0.01] * 9, skill_drawdown_penalty=0, turnover_cost=0)
    assert not result.pass_gate


def test_feedback_does_not_accept_nonfinite_series():
    with pytest.raises(ValueError, match="finite"):
        evaluate_feedback(current_reliability=0.8, signal_values=[1, math.nan],
                          future_net_returns=[0.01, 0.02], skill_drawdown_penalty=0, turnover_cost=0)


def test_research_promotion_gates_are_pure_and_require_evidence():
    result = evaluate_promotion_formula(current_mode="RESEARCH", target_mode="SHADOW")
    assert not result.pass_gate
    result = evaluate_promotion_formula(current_mode="RESEARCH", target_mode="SHADOW",
                                        validation_manifest_pass=True)
    assert result.pass_gate
    assert result.evidence["next_mode"] == "SHADOW"


@pytest.mark.parametrize("quantity", [0, -1, 20])
def test_sell_quantity_must_be_positive_and_owned(quantity):
    result = evaluate_trade_permission(mode="PAPER", direction="SELL", data_integrity_pass=True,
                                       owned_qty=10, sell_qty=quantity)
    assert not result.pass_gate
