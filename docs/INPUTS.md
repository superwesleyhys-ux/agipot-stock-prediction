# 输入与时间约定

`analyze_stock(payload)` 与 `agipot-predict --input input.json` 接受相同字段。CLI 严格拒绝 JSON 中的 NaN/Infinity；每个模块还会执行各自的数据校验。

## 基本输入

| 字段 | 类型 | 含义 |
|---|---|---|
| `symbol` | string，必需 | 股票代码，统一为大写；末尾 `.US` 会移除 |
| `as_of` | ISO datetime，必需 | 分析截止时间，必须带 UTC 偏移或 `Z` |
| `daily_rows` | array，必需 | 每个交易日一个 OHLCV 记录，空列表返回数据不足报告 |
| `data_source` | string，可选 | 调用方提供的数据来源标签，默认 `caller_supplied` |

日线最少包含 `date`（YYYY-MM-DD）、正数 `close`；建议提供 `open`、`high`、`low`、`volume`、`adjusted_close`。缺少 OHLC 时，个股研判会用 close 补齐；这不是完整的真实 OHLC。评分模块需要正收盘价和非负成交量，缺少成交量按零处理，流动性门槛会阻断。统一 API 中重复纽约日期会被评分模块拒绝。

个股研判按 XNYS 日历和实际收盘时间过滤未完成日线，包括节假日与半日交易。评分模块更保守：只使用纽约 `as_of` 日期之前的日线，并为趋势特征跳过最近 5 个已完成观察；它的周期数按调用方提供的日线记录计数。输入应只包含有效交易日。两者末端日期可能不同，输出中分别披露。

个股研判的趋势分析使用 `adjusted_close`；评分模块直接使用 `close`，调用方应提供与成交量口径一致、已经处理拆股的 close 序列。不要把不同复权口径混合。传入当前下载的复权历史不能自动保证历史时点可见性。

## 分钟线与逐笔成交

`minute_rows` 为数组，包含 ISO 或 epoch `timestamp`、OHLCV。时间字符串建议明确带时区；个股研判中无时区值按 UTC 处理。分钟线只是短线参考，不能满足秒级数据门槛。每条 bar 的 timestamp 应代表已经完成的观察；开盘时间戳需要调用方先移至完成时刻或排除未完成 bar。

`tick_payload` 使用与原解析器一致的列式格式：

```json
{
  "ts": [1764599400000, 1764599401000],
  "price": [100.0, 100.1],
  "shares": [100, 200],
  "seq": [1, 2],
  "sl": ["", ""],
  "mkt": ["EXAMPLE", "EXAMPLE"]
}
```

`ts` 为 Unix 秒或毫秒；`price` 必需，其余列可选。解析器过滤会话区间外、重复 sequence 和被排除的成交条件；成交条件规则来自原模块，应按数据供应商定义核对。成交记录本身不能补出历史最佳买卖报价 NBBO。

## 两个基本面入口

`fundamentals` 用于单票研判，接受应用内部整理后的季度报表结构，不是供应商原始响应：

```json
{
  "ok": true,
  "financialRows": [
    {"revenue": 1200000, "grossMargin": 45, "freeCashFlow": 180000, "netIncome": 150000},
    {"revenue": 1150000, "grossMargin": 44, "freeCashFlow": 170000, "netIncome": 145000},
    {"revenue": 1100000, "grossMargin": 44, "freeCashFlow": 160000, "netIncome": 140000},
    {"revenue": 1050000, "grossMargin": 43, "freeCashFlow": 150000, "netIncome": 130000},
    {"revenue": 1000000, "grossMargin": 42, "freeCashFlow": 140000, "netIncome": 120000}
  ]
}
```

记录由新到旧排列，索引 4 作为一年前季度比较；`grossMargin` 是百分数（45 表示 45%）。收入、利润、现金流应使用相同货币和单位。调用方必须先排除截止日尚未披露的季度。

`factor_fundamentals` 用于因子评分，是平铺数值映射，例如 `{"roic": 0.18, "gross_margin": 0.45, "pe_ratio": 25}`。收益率、利润率用小数，倍数用比率，未知字段拒绝。完整字段见 [scoring/api.py](../src/agipot_stock_prediction/scoring/api.py) 中 `FUNDAMENTAL_KEYS` 和 [评分说明](SCORING.md)。缺少基本面会使用明确标记的价格质量代理，而个股研判的基本面门槛仍保持阻断。

## 日内模型

`intraday_features` 为有限数值字典，例如 `ret_5m`、`ret_15m`、`ret_60m`、`relative_volume`、`spread_bps`。收益率用小数，价差用基点。统一入口输出规则 alpha；如通过 Python 的 `model=` 传入已训练模型，也输出在线 alpha。

`AdaptiveEdgeModel.learn(features, realized_return_bps=...)` 不管理时间：调用方必须等待收益标签完整可见后再学习，不能在同一个时点先使用未来收益训练再预测。CLI 不自动训练模型。独立 `IntradayBar` 要求带时区、合法 OHLC 和成对未交叉 bid/ask；特征提取要求单一股票、严格递增的时间顺序。
