from datetime import datetime, timedelta, timezone
import math
import pytest

from agipot_stock_prediction.intraday import (
    AdaptiveEdgeModel, IntradayBar, MicrostructureFeatureState,
    compute_microstructure_features, family_alpha,
)


def bars():
    start = datetime(2025, 12, 1, 14, 30, tzinfo=timezone.utc)
    return [IntradayBar(start + timedelta(minutes=i), "SYNTH", 100+i/10,
                       101+i/10, 99+i/10, 100.2+i/10,
                       1000 if i % 8 else 50000) for i in range(80)]


def test_streaming_and_batch_features_agree():
    state = MicrostructureFeatureState()
    history = bars()
    for i, bar in enumerate(history):
        assert state.update(bar) == pytest.approx(compute_microstructure_features(history[:i+1]))


def test_history_order_is_required():
    with pytest.raises(ValueError, match="timestamp"):
        compute_microstructure_features(list(reversed(bars())))


def test_nonfinite_inputs_do_not_become_valid_scores():
    with pytest.raises(ValueError, match="finite"):
        family_alpha("intraday_momentum", {"ret_5m": math.nan})
    with pytest.raises(ValueError, match="finite"):
        AdaptiveEdgeModel().edge_bps({"ret_5m": math.inf})


def test_model_has_warmup_and_learns_signed_edge():
    model = AdaptiveEdgeModel(min_samples=3, edge_floor_bps=0)
    features = {"ret_5m": 0.002}
    assert model.edge_bps(features) == 0
    for _ in range(3):
        model.learn(features, realized_return_bps=10)
    assert 0 < model.edge_bps(features) <= 50
    assert model.edge_bps({"ret_5m": -0.002}) < 0


def test_bar_rejects_crossed_quotes():
    with pytest.raises(ValueError, match="uncrossed"):
        IntradayBar(datetime.now(timezone.utc), "SYNTH", 100, 101, 99, 100, 1000, bid=101, ask=100)


def test_new_york_session_date():
    bar = IntradayBar(datetime(2025, 12, 2, 0, 0, tzinfo=timezone.utc), "SYNTH", 100, 101, 99, 100, 1)
    assert bar.session.isoformat() == "2025-12-01"


def test_cost_rejects_nonfinite_borrow_fee():
    from agipot_stock_prediction.intraday.intraday_cost_model import compute_intraday_cost
    with pytest.raises(ValueError, match="borrow_fee_annualized must be finite"):
        compute_intraday_cost(predicted_return_bps=100, spread_bps=1,
                              order_notional=100, dollar_volume_window=100000,
                              borrow_fee_annualized=math.nan)
