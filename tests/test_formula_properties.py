"""Independent oracles and generated numerical boundary combinations."""
import math
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st
from agipot_stock_prediction.formulas.ensemble import evaluate_ensemble
from agipot_stock_prediction.harnesses.formula_contract import FormulaContractHarness
from agipot_stock_prediction.research.cost_model import apply_cost_to_return, estimate_transaction_cost

finite = st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False)
costs = st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)


@given(gross=finite, first=costs, second=costs)
def test_nondecreasing_cost_cannot_raise_net_return(gross, first, second):
    low, high = sorted((first, second))
    assert apply_cost_to_return(gross_return=gross, cost_bps=high) <= apply_cost_to_return(gross_return=gross, cost_bps=low)
    assert apply_cost_to_return(gross_return=gross, cost_bps=low) == gross - low/10000


@given(scores=st.lists(finite, min_size=3, max_size=7), cap=st.floats(min_value=0.34, max_value=1, allow_nan=False, allow_infinity=False))
def test_ensemble_simplex_and_cap(scores, cap):
    values = {str(i): value for i, value in enumerate(scores)}
    result = evaluate_ensemble(skill_scores=values, skill_reliabilities=values, max_skill_weight=cap)
    weights = result.evidence["weights"]
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    assert all(0 <= value <= cap + 1e-12 for value in weights.values())
    assert 0 <= result.score <= 1


@given(notional=st.floats(1, 1e8), adv=st.floats(1, 1e10), spread=st.floats(0, 100), fee=st.floats(0, 100))
def test_cost_matches_independent_units_oracle(notional, adv, spread, fee):
    result = estimate_transaction_cost(notional=notional, median_daily_dollar_volume=adv, spread_bps=spread, commission=fee)
    expected_bps = spread/2 + 10*math.sqrt(notional/adv)
    assert result.total_cost == pytest.approx(fee+notional*expected_bps/10000)
    assert result.total_cost_bps == pytest.approx(expected_bps+fee/notional*10000)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True])
def test_nonfinite_or_bool_cannot_enter_costs(bad):
    with pytest.raises(ValueError):
        apply_cost_to_return(gross_return=bad, cost_bps=0)


@pytest.mark.parametrize("field", ["spread_bps", "commission", "spread_capture_ratio", "impact_coefficient"])
def test_negative_cost_parameter_rejected(field):
    kwargs = {"notional": 100.0, "median_daily_dollar_volume": 10000.0, "spread_bps": 1.0, field: -1.0}
    with pytest.raises(ValueError):
        estimate_transaction_cost(**kwargs)


def test_zero_cost_is_preserved_and_negative_fee_rejected():
    assert apply_cost_to_return(gross_return=0.015, cost_bps=0) == 0.015
    with pytest.raises(ValueError):
        apply_cost_to_return(gross_return=0.015, cost_bps=-1)
    for notional, adv in [(0,1), (1,0), (-1,1), (1,-1)]:
        with pytest.raises(ValueError):
            estimate_transaction_cost(notional=notional, median_daily_dollar_volume=adv, spread_bps=0)
    assert FormulaContractHarness().run()["status"] == "PASS"


def test_default_and_zero_cost_components_are_public_contract():
    default = estimate_transaction_cost(notional=100, median_daily_dollar_volume=10000, spread_bps=2)
    assert default.notional == 100
    assert default.commission == 0
    assert default.spread_cost_bps == 1
    assert default.impact_bps == 1
    assert default.total_cost == pytest.approx(.02)
    assert default.total_cost_bps == pytest.approx(2)
    zero = estimate_transaction_cost(notional=100, median_daily_dollar_volume=10000, spread_bps=0,
                                     commission=0, spread_capture_ratio=0, impact_coefficient=0)
    assert zero.spread_cost_bps == zero.impact_bps == zero.total_cost == zero.total_cost_bps == 0


@pytest.mark.parametrize("field", ["notional", "median_daily_dollar_volume", "spread_bps", "commission", "spread_capture_ratio", "impact_coefficient"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, None])
def test_cost_parameter_errors_identify_the_rejected_field(field, bad):
    kwargs = {"notional": 100, "median_daily_dollar_volume": 10000, "spread_bps": 2, field: bad}
    with pytest.raises(ValueError, match="cost\\." + field):
        estimate_transaction_cost(**kwargs)


@pytest.mark.parametrize("field", ["gross_return", "cost_bps"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, None])
def test_return_parameter_errors_identify_the_rejected_field(field, bad):
    with pytest.raises(ValueError, match="cost\\." + field):
        apply_cost_to_return(**{"gross_return": .01, "cost_bps": 2, field: bad})
