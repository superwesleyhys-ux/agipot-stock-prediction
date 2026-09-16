"""Run after installing the package: python examples/demo.py."""
import json
from agipot_stock_prediction import analyze_stock
from agipot_stock_prediction.demo import demo_input

report = analyze_stock(demo_input())
print(json.dumps({
    "symbol": report["symbol"],
    "index": report["distillation"]["specific_stock_index"],
    "scores": report["scores"]["scores"],
}, ensure_ascii=False, indent=2, allow_nan=False))
