# AGIPOT Stock Prediction & Scoring

An offline Python toolkit extracted from AGIPOT: multi-horizon stock research, transparent factor scoring, an online feature/return model, and validation utilities. MIT licensed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
agipot-predict --demo --output demo-report.json
python -m pytest -q
```

Python 3.11+ is required. The demo generates fictional data deterministically. No market dataset, trained weights, account configuration, provider clients or broker execution is included.

The unified `analyze_stock(payload)` API accepts a symbol, timezone-aware `as_of` and `daily_rows`, with optional minute bars, trades, fundamentals and intraday features. It returns three separately defined outputs: multi-horizon `distillation`, factor `scores`, and `intraday` alpha. See [inputs](INPUTS.md), [scoring](SCORING.md) and [provenance](PROVENANCE.md).

The scoring package exposes trend, momentum and quality/value factors together with quality proxies, risk overlays and allocation primitives. The formulas package contains auditable quality, moat, valuation, liquidity, transaction-cost, data-quality, ensemble and risk formulas. Validation utilities construct chronological windows, rank trial results and compute explicitly labeled overfitting heuristics.

Scores and confidence fields are not calibrated probabilities. Online edge estimates are not demonstrated tradable expected returns. The historical holdout is descriptive research and does not replace independently tested walk-forward performance. Fundamentals, adjustments, observation completion times and data rights remain caller responsibilities. PBO/DSR proxies are not formal statistical estimators. Tests verify software behavior, not profitability.
