"""AGIPOT stock research and transparent scoring tools."""

__version__ = "0.2.0"


def analyze_stock(payload, *, model=None):
    """Run the offline research pipeline; see :mod:`agipot_stock_prediction.api`."""
    from .api import analyze_stock as analyze
    return analyze(payload, model=model)


__all__ = ["analyze_stock", "__version__"]
