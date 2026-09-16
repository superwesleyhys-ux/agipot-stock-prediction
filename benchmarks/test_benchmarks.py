from agipot_stock_prediction.intraday import AdaptiveEdgeModel


def test_online_edge_prediction(benchmark):
    model = AdaptiveEdgeModel(min_samples=10)
    features = {"ret_5m": 0.003, "relative_volume": 1.5}
    for _ in range(50):
        model.learn(features, realized_return_bps=3)
    benchmark(model.edge_bps, features)
