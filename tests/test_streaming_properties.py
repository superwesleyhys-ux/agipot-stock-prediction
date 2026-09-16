from datetime import datetime, timedelta, timezone
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st
from agipot_stock_prediction.intraday import IntradayBar, MicrostructureFeatureState, compute_microstructure_features


@given(st.lists(st.tuples(st.floats(1, 1000, allow_nan=False), st.floats(0, 1e6, allow_nan=False)), min_size=1, max_size=85))
@settings(max_examples=35, deadline=None)
def test_streaming_matches_batch_across_sessions(points):
    state, history = MicrostructureFeatureState(), []
    start = datetime(2025, 11, 28, 17, tzinfo=timezone.utc)
    for i, (price, volume) in enumerate(points):
        stamp = start + timedelta(minutes=i, days=i//40)
        bar = IntradayBar(stamp, "SYNTH", price, price, price, price, volume)
        history.append(bar)
        assert state.update(bar) == pytest.approx(compute_microstructure_features(history), rel=1e-10, abs=1e-10)
