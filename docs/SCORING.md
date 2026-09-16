# 股票因子评分、融合与研究仓位

`score_stock` 将调用者提供的日线输入依次转换为趋势特征、质量代理，再计算 `trend`、`momentum`、`value` 三个独立分数。返回值包含分数、资格门槛结果、阻断原因和中间特征，能够使用 `json.dumps(result, allow_nan=False)` 序列化。

```python
from agipot_stock_prediction.demo import demo_input
from agipot_stock_prediction.scoring import score_stock

data = demo_input()  # 项目自带的确定性虚构数据
result = score_stock(
    data["daily_rows"],
    symbol=data["symbol"],
    as_of=data["as_of"],
    fundamentals=data["factor_fundamentals"],
)
for name, signal in result["scores"].items():
    print(name, signal["score"], signal["eligible"], signal["blocking_reasons"])
```

统一入口 `analyze_stock(payload)` 将 `payload.factor_fundamentals` 传入上面的 `fundamentals` 参数。统一入口的 `payload.fundamentals` 是另一份供个股蒸馏模块使用的嵌套数据，两者不自动转换。

## 日线特征如何形成

- 按纽约日期保留严格早于 `as_of` 日期的行。同日收盘数据也排除；带时区时间先转换到纽约。直接调用 `score_stock` 时，日期字符串和无时区时间按纽约时间解释。
- 趋势收益、波动率、成交额及该组特征的 `latest_close`，还会跳过最近 **5 条完整日线**。`latest_close` 因此是趋势特征窗口的末值。
- `return_20`、`return_60`、`return_120` 使用对应观察点之间的简单价格收益。窗口中的“日”表示输入的每日观察条数；评分函数不补齐缺失交易日，也不自行调整拆股、分红或成交量。
- 年化波动率使用最近 20 个对数收益的总体标准差乘以 `sqrt(252)`；日成交额使用 `close × volume` 的最近 20 条中位数。
- 质量模块的 `drawdown_120` 使用最近 120 条完整日线的最大回撤，**没有额外跳过 5 条**；最长不足 120 条时使用可用窗口。质量模块至少要求 60 条完整日线。

三类评分均继承特征阶段的阻断原因。完整的 120 条收益窗口加上端点及 5 条跳过窗口，需要至少 **126 条完整日线**。非有限数、非正收盘价、负成交量或同一纽约日期重复行会抛出 `ValueError`；有效数据不足时返回阻断原因，不伪造缺失历史。`volume` 缺省为 0，通常会触发流动性门槛。

## 三类分数和默认门槛

| 分数 | 范围 | 含义 |
| --- | --- | --- |
| `trend` | −2 到 2 | 三个周期收益相对于历史波动率的强度，0 为中性 |
| `momentum` | 0 到 1 | 动量、趋势一致性与风险质量的加权值，扣除过热惩罚 |
| `value` | 0 到 1 | 质量、护城河、估值与安全边际代理的加权值 |

`score` 和 `eligible` 应同时读取。未通过门槛的标的仍保留诊断分数；高分不能覆盖缺数据或风险阻断。这里的分数不是预期收益率，名称中的 Buffett 指启发式质量价值规则，不代表任何个人认可或其真实投资模型。

### 趋势 `trend`

对 `L ∈ {20, 60, 120}`：

```text
component_L = clip(return_L / (annualized_volatility × sqrt(L / 252)), -2, 2)
trend_score = mean(component_20, component_60, component_120)
```

三周期等权。默认资格条件为：全部窗口可用、波动率有限且大于 0、日成交额中位数至少 **5,000 万美元**、分数严格大于 0，并且没有继承的阻断原因。

### 动量 `momentum`

线性归一化后的各项截断到 `[0, 1]`：

| 项目 | 从 0 分到 1 分的区间 | 权重 |
| --- | --- | --- |
| 120 条收益 | 2% → 30% | 27% |
| 60 条收益 | 0% → 18% | 25% |
| 20 条收益 | −3% → 8% | 12% |
| 趋势一致性 | 三个收益中为正的比例 | 14% |
| 波动率质量 | 年化波动率 60% → 12% | 7% |
| 回撤质量 | 最大回撤 40% → 5% | 7% |
| 流动性质量 | 日成交额 2,000 万 → 2.5 亿美元 | 4% |
| `quality_proxy` | 见下文 | 4% |

加权后减去 `0.12 × clip((return_20 - 0.16) / 0.16, 0, 1)`，最后截断到 `[0, 1]`。这意味着 20 条收益超过 16% 时逐渐受到过热惩罚，惩罚上限为 0.12。

默认门槛：60 条收益至少 **2%**，120 条收益至少 **4%**，20 条收益至少 **−6%**，年化波动率不超过 **80%**，120 条窗口回撤不超过 **35%**，日成交额中位数至少 **2,000 万美元**，最终分数至少 **0.50**，且没有继承的阻断原因。

### 质量价值 `value`

| 项目 | 权重 |
| --- | --- |
| 综合质量 `quality_proxy` | 26% |
| 业务质量 `business_quality_proxy` | 16% |
| 财务强度 `financial_strength_proxy` | 14% |
| 护城河代理 `moat_proxy` | 12% |
| 安全边际代理 `margin_of_safety_proxy` | 18% |
| 估值代理 `valuation_proxy` | 8% |
| 趋势一致性 `trend_consistency` | 6% |

默认门槛：综合质量至少 **0.55**，安全边际代理至少 **0.40**，年化波动率不超过 **65%**，120 条窗口回撤不超过 **45%**，日成交额中位数至少 **1,000 万美元**，最终分数至少 **0.52**，且没有继承的阻断原因。

## 质量代理和基本面缺口

全部归一化因子都截断到 `[0, 1]`。加权平均只使用可用分项，并按可用权重重新归一化；真实的 0 分保留为 0，不当作缺失值。

| 代理 | 组成及内部权重 |
| --- | --- |
| 盈利能力 | ROIC 35%、ROE 15%、自由现金流利润率 20%、营业利润率 15%、自由现金流转换率 15% |
| 稳定性 | 毛利率稳定性 30%、盈利稳定性 35%、收入稳定性 20%、波动率质量 15% |
| 财务强度 | 债务权益比 30%、净债务/自由现金流 25%、净债务/EBITDA 20%、利息覆盖倍数 15%、股份稀释率 10% |
| 护城河 | 毛利率 25%、毛利率稳定性 25%、ROIC 30%、盈利稳定性 20% |
| 价格质量 | 趋势一致性 35%、波动率质量 30%、回撤质量 25%、流动性质量 10% |

综合外部质量由盈利能力 **38%**、稳定性 **27%**、财务强度 **22%**、护城河 **13%** 组成；最终 `quality_proxy` 将它与价格质量按 **65% / 35%** 混合。基本面完整时仍然包含价格信息。

完全没有基本面时，“稳定性”只剩波动率质量，其他外部分项不可用。因此当前实现退化为：

```text
quality_proxy = 0.65 × volatility_quality + 0.35 × price_quality
```

此时业务质量、财务强度和护城河代理也会回退到综合质量。它们是价格行为的代理值，不能解释成已经测量了企业盈利能力、偿债能力或护城河。

估值代理使用股东收益率 **30%**、自由现金流收益率 **25%**、盈利收益率 **15%**、市价/自由现金流 **15%**、市盈率 **10%**、EV/EBIT **5%**；全部缺失时取 **0.50**。显式安全边际和内在价值差距先归一化，再按 **55% / 45%** 合成安全边际代理；两者都缺失时，使用 **60% 估值代理 + 40% 回撤质量**。因此 `margin_of_safety_proxy` 不等于直接测算出的折价率。

`external_fundamental_coverage` 统计 14 个质量类字段的覆盖比例；估值和安全边际字段不计入该比例。`uses_price_quality_proxies` 表示这 14 个字段是否有缺口，不能据此推断价格因子是否参与计算，也不能推断估值数据是否齐全。

### 基本面单位

直接调用 `score_stock(..., fundamentals=...)` 时使用平坦映射；统一入口使用 `factor_fundamentals`。值必须为有限数，字段必须属于公开的 [`FUNDAMENTAL_KEYS`](../src/agipot_stock_prediction/scoring/api.py) 集合。

| 字段组 | 单位与例子 |
| --- | --- |
| `roic`、`roe`、`gross_margin`、`operating_margin`、`fcf_margin` | 小数比例；20% 写 `0.20` |
| `owner_earnings_yield`、`fcf_yield`、`earnings_yield` | 小数收益率；5% 写 `0.05` |
| `share_dilution`、`margin_of_safety`、`intrinsic_value_gap` | 小数比例；8% 写 `0.08`；安全边际越高评分越高 |
| `fcf_conversion` | 转换比例；100% 写 `1.0` |
| `debt_to_equity`、`net_debt_to_fcf`、`net_debt_to_ebitda`、`interest_coverage` | 比值或倍数；3 倍写 `3.0` |
| `price_to_fcf`、`pe_ratio`、`ev_to_ebit` | 估值倍数；25 倍写 `25.0` |
| `gross_margin_stability`、`earnings_stability`、`revenue_stability` | 调用者定义的稳定性指标，越高越稳定；本模型将 `0.40 → 0.90` 映射为 `0 → 1`，不会从原始财务报表自行计算 |

字段列表也可通过 `from agipot_stock_prediction.scoring import FUNDAMENTAL_KEYS` 读取。全部阈值和公式以 [`quality_proxy.py`](../src/agipot_stock_prediction/scoring/quality_proxy.py) 为准。调用者负责保证财务值在 `as_of` 时已经公开；本接口没有财报发布日期验证或单位自动推断。

## `confidence` 不是概率

- 趋势信号的底层置信值按数据及过滤原因取 0 或 1。趋势非正而被判为不合格时，仍可能保留数据完整度意义上的 1；应以 `eligible` 和 `blocking_reasons` 判断资格。
- 动量合格时取 `0.72 + 0.18 × 趋势一致性 + 0.10 × 综合质量`，不合格时为 0。
- 质量价值合格时取 `0.58 + 0.32 × 基本面覆盖率 + 0.10 × 趋势一致性`，不合格时为 0。

这些规则没有做胜率校准，不能把 `confidence=0.8` 解释为上涨概率 80%。在多因子融合中使用的 `skill_reliabilities` 应来自独立验证或明确的研究假设，不能自动把上述启发式值当成样本外可靠性。

## 三因子归一化后融合

[`evaluate_ensemble`](../src/agipot_stock_prediction/formulas/ensemble.py) 对调用者提供的分数使用 `[0, 1]` 范围，以可靠性 softmax 形成权重，并限制单因子权重。下面是可直接运行的例子，显式把趋势 `[-2, 2]` 线性映射为 `[0, 1]`；动量、价值已在该范围内。

```python
import json
from agipot_stock_prediction.demo import demo_input
from agipot_stock_prediction.scoring import score_stock
from agipot_stock_prediction.formulas.ensemble import evaluate_ensemble

data = demo_input()
result = score_stock(
    data["daily_rows"],
    symbol=data["symbol"],
    as_of=data["as_of"],
    fundamentals=data["factor_fundamentals"],
)
signals = result["scores"]
normalized = {
    "trend": (signals["trend"]["score"] + 2.0) / 4.0,
    "momentum": signals["momentum"]["score"],
    "value": signals["value"]["score"],
}
assert all(0.0 <= value <= 1.0 for value in normalized.values())
ensemble = evaluate_ensemble(
    skill_scores=normalized,
    # 演示假设，不是回测得出的可靠性，也不是上涨概率。
    skill_reliabilities={"trend": 0.70, "momentum": 0.65, "value": 0.60},
    pass_threshold=0.70,
    temperature=0.25,
    max_skill_weight=0.40,
)
print(json.dumps({
    "ensemble": ensemble.canonical_payload(),
    "all_sources_eligible": all(signal["eligible"] for signal in signals.values()),
    "research_gate": ensemble.pass_gate and all(
        signal["eligible"] for signal in signals.values()
    ),
}, ensure_ascii=False, indent=2, allow_nan=False))
```

三因子配合 40% 的单因子上限可行，因为 `3 × 0.40 >= 1`。若删减因子，需要同时满足“因子数量 × 上限至少为 1”。可靠性为 0 仍可能分到 softmax 权重；融合函数也不会读取原评分的 `eligible`。示例因此单独保留来源门槛，融合不能解除上游阻断。

趋势线性映射只是便于演示的量纲对齐，没有证明三种分数具有相同的预测含义；它们也共享价格特征，不能视为独立证据。融合结果的 `confidence` 是输入可靠性的加权平均，同样不是经校准的盈利概率。

## 低层风险和仓位 API

这些组件接收本地 `SkillContext`，返回包含 `.payload` 的 `RawSkillOutput`。只计算研究状态或目标权重，不连接账户或提交订单。`score_stock` 本身不会自动调用下面的组合分配组件。

| 组件 | 默认行为 |
| --- | --- |
| `MinimumPositiveSleevesRegime` | 至少 3 个信号标的 `eligible=True` 才为 `TREND_ON`；不足时 `DEFENSIVE_CASH`、`risk_scalar=0`。计数对象是标的，不是同一标的的三个分数 |
| `InverseVolatilityAllocator` | 对合格且分数、波动率为正的信号，以 `score / volatility` 为原始权重；默认总目标 70%、单标的上限 5%；**不自动读取 regime** |
| `ConvictionAllocatorSkill` | 使用 `max(score − score_floor, 0) × confidence`，默认再除以至少 8% 的波动率；默认总目标 60%、单标的上限 10%；总目标乘以 `regime.risk_scalar` |
| `BuffettRiskOverlaySkill` | 按正仓位加权质量、安全边际和波动率；默认质量至少 0.55、安全边际至少 0.40、波动率不超过 55%；返回缩放系数和检查项，不直接修改仓位 |

超出单标的上限的预算会向其余可用标的重分配；标的不足时剩余部分留现金。风险 overlay 每个失败检查的缩放因子最低为 0.25，多个因子相乘，最终 multiplier 可以低于 0.25。它是缩放机制，不是账户硬性熔断器。

下面用三个虚构标的展示 regime → conviction allocation → risk overlay 的组合方式；研究信号数值仅用于演示接口：

```python
from datetime import datetime, timezone
from agipot_stock_prediction.scoring import (
    BuffettRiskOverlaySkill,
    ConvictionAllocatorSkill,
    MarketDataSnapshot,
    MinimumPositiveSleevesRegime,
    SkillBinding,
    SkillContext,
    SkillVersion,
)

state = {
    "signals": {
        "SYNTH_A": {"eligible": True, "score": 0.8, "confidence": 0.7, "annualized_volatility": 0.2},
        "SYNTH_B": {"eligible": True, "score": 0.7, "confidence": 0.7, "annualized_volatility": 0.3},
        "SYNTH_C": {"eligible": True, "score": 0.6, "confidence": 0.7, "annualized_volatility": 0.25},
    },
    "features": {
        symbol: {
            "quality_proxy": 0.65,
            "margin_of_safety_proxy": 0.50,
            "annualized_volatility": signal["annualized_volatility"],
        }
        for symbol, signal in {
            "SYNTH_A": {"annualized_volatility": 0.2},
            "SYNTH_B": {"annualized_volatility": 0.3},
            "SYNTH_C": {"annualized_volatility": 0.25},
        }.items()
    },
}
snapshot = MarketDataSnapshot(as_of=datetime(2025, 1, 2, tzinfo=timezone.utc))
version = SkillVersion()

def context(config=None):
    return SkillContext(snapshot=snapshot, state=state, binding=SkillBinding(config=config or {}))

state["regime"] = MinimumPositiveSleevesRegime(version).execute(context()).payload
portfolio = ConvictionAllocatorSkill(version).execute(context()).payload
state["weights"] = portfolio["weights"]
overlay = BuffettRiskOverlaySkill(version).execute(context()).payload
scaled_weights = {
    symbol: weight * overlay["multiplier"]
    for symbol, weight in portfolio["weights"].items()
}
print({"weights": scaled_weights, "cash_weight": 1.0 - sum(scaled_weights.values())})
```

在多股票研究中，为每个标的选择一种一致的评分及其对应置信值组成 `state["signals"]`；若采用融合分数，需先完成量纲归一化并保留上游门槛。配置项通过 `SkillBinding(config={...})` 传入，具体参数由各组件源码定义。

## 源码和验证

实现位于 [`scoring/`](../src/agipot_stock_prediction/scoring/)，边界测试位于 [`tests/test_scoring.py`](../tests/test_scoring.py)，覆盖历史不足、未来及同日行排除、纽约日期、非有限值、真实零分、分配上限和防御现金状态。提取后的本地契约不依赖原应用服务器、数据库、券商或认证系统。
