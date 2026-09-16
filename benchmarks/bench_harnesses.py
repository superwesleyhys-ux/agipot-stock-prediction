from agipot_stock_prediction.intraday import AdaptiveEdgeModel


class OnlineEdge:
    def setup(self):
        self.model = AdaptiveEdgeModel(min_samples=1)
        self.features = {"ret_5m": 0.001, "relative_volume": 2.0}
        self.model.learn(self.features, realized_return_bps=3)

    def time_predict(self):
        self.model.edge_bps(self.features)

    def peakmem_predict(self):
        self.model.edge_bps(self.features)
