# AGIPOT Stock Prediction & Scoring

中文 · [English](docs/README_EN.md) · [Scoring reference](docs/SCORING.md) · [Source provenance](docs/PROVENANCE.md)

从 AGIPOT 提取的股票研究与评分工具箱。把单票多周期分析、在线收益边际模型、趋势／动量／质量价值评分，以及数据、成本、风险、验证公式整理到同一个 Python 包中。

**这是可解释的统计与规则研究代码。分数不是涨跌概率，示例不是收益验证，没有附带训练权重或历史盈利承诺。**

## 包含什么

| 模块 | 内容 |
|---|---|
| `distillation` | 成交秒线、日线、周月线研判，25%／35%／40% 多周期指数，历史相似状态与时间留出统计 |
| `intraday` | 在线 `AdaptiveEdgeModel`、日内／横截面信号、VWAP／量价／波动特征、交易成本估算 |
| `scoring` | 趋势特征、趋势融合、动量龙头、质量代理、质量价值、风险覆盖、组合权重分配 |
| `formulas` | 质量、护城河、所有者收益、内在价值、安全边际、流动性、融合权重、仓位、成本、反馈与数据质量门槛 |
| `validation` | 时间前推窗口、试验排序与过拟合启发式检查 |

包内通过本地 Python 对象和 JSON 输入运行。数据获取适配器、账户、密钥、券商连接与自动下单不属于这个发布包。

## 快速运行

要求 Python 3.11 或以上。

```bash
git clone https://github.com/superwesleyhys-ux/agipot-stock-prediction.git
cd agipot-stock-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
agipot-predict --demo --output demo-report.json
python -m pytest -q
```

Windows 使用 `.venv\Scripts\activate` 激活环境。演示数据为固定生成的虚构行情，不是任何真实股票的历史记录。

## 使用自己的数据

```bash
agipot-predict --input input.json --output report.json
```

最小输入：

```json
{
  "symbol": "EXAMPLE",
  "as_of": "2025-12-01T21:00:00+00:00",
  "daily_rows": [
    {"date": "2025-11-28", "open": 100, "high": 102, "low": 99,
     "close": 101, "adjusted_close": 101, "volume": 1000000}
  ]
}
```

一天的数据只会演示数据不足时的阻断状态。有效研究需要足够历史：日线模块至少 252 个点，趋势评分默认回看 20／60／120 日并跳过最近 5 日；完整历史门槛为 8 年。缺少成交秒线时秒级分数为空，不用分钟线代替。

可选字段：

- `minute_rows`: 当日分钟 OHLCV，使用带时区的 `timestamp`；`as_of` 之后的数据被排除。
- `tick_payload`: 原始成交的列式数组对象，字段为 `ts`、`price`、`shares` 等，详见 [输入说明](docs/INPUTS.md)。
- `fundamentals`: 个股研判用的嵌套基本面对象。
- `factor_fundamentals`: 因子评分用的平铺数值字典，两个基本面接口的字段含义详见输入说明。
- `intraday_features`: 单个已完成观测的日内特征字典。未提供时此模块显示 `NOT_PROVIDED`。

Python 接口：

```python
from agipot_stock_prediction import analyze_stock
from agipot_stock_prediction.scoring import score_stock
from agipot_stock_prediction.intraday import AdaptiveEdgeModel

report = analyze_stock({
    "symbol": "EXAMPLE",
    "as_of": "2025-12-01T21:00:00+00:00",
    "daily_rows": [],
})

model = AdaptiveEdgeModel(min_samples=240)
# 只有持有期已经结束、收益标签已经可见时才调用 learn。
model.learn({"ret_5m": 0.002}, realized_return_bps=8.0)
edge_bps = model.edge_bps({"ret_5m": 0.001})
```

统一报告保留三类结果：`distillation`、`scores`、`intraday`。不同量纲的分数不会直接相加。要构建自己的融合规则，可使用 `formulas.ensemble.evaluate_ensemble` 并显式提供已归一化的分数和可靠性。

## 研究边界

- 指数是启发式方向评分；`confidence` 表示规则定义的数据覆盖／合格程度，并非校准后的成功概率。
- `edge_bps` 是简单在线相关性模型的输出，未证明是无偏或可交易的预期收益。
- 时间留出统计仍受重叠标签、当前行情状态筛选及多重试验影响。没有替代独立的滚动样本外验证。
- 基本面必须使用当时已披露的数据；复权、幸存者偏差、数据授权由数据提供者处理。
- PBO／DSR 名称带 `proxy` 的输出仅是启发式指标，不是正式的概率过拟合或 Deflated Sharpe 估计。
- 每条 intraday bar 按一个采样步计数；`ret_5m` 等名称仅在完整、等间隔的一分钟数据下才对应分钟数。VWAP 使用 bar close 近似成交均价；默认价差和盘口字段包含代理假设。

## 开发与许可证

```bash
python -m pytest -q
python -m ruff check src tests
python -m build
```

MIT，允许使用、修改和再分发。第三方依赖及自行提供的行情数据适用各自许可证。[发布改动与来源](docs/PROVENANCE.md)说明从原模块提取和修正的范围。
