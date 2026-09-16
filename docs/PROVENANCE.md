# 来源与发布适配

该公开包由项目所有者授权，从 AGIPOT 的股票研究与评分代码提取。源代码基线为 `176809b480f8f0833a5977b4d3c0f7488bb0b41b`。逐文件源路径、SHA-256 和新路径记录在 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。清单中的 hash 标识提取前源文件，不是修改后公开文件的 hash。

## 发布范围

| 原模块 | 公开位置 |
|---|---|
| `specific_stock_distillation.py` | `distillation.py`：保留纯计算函数，移除异步行情获取、提供商客户端及市场数据缓存 |
| `us_market_calendar.py` | `calendar.py`：保留 XNYS 交易日及实际开收盘时间 |
| `intraday_factory/signal_library.py` | `intraday/signal_library.py`：在线模型、日内及横截面信号 |
| `intraday_factory/microstructure_features.py` | `intraday/microstructure_features.py`：批量及增量特征 |
| `intraday_factory/intraday_cost_model.py` | `intraday/intraday_cost_model.py`：成本研究公式 |
| `skills/builtins/` 中 11 个相关模块 | `scoring/`：特征、质量、趋势、动量、价值、风险、分配及数学工具 |
| `formulas/` | `formulas/`：完整纯公式层及最小本地依赖 |
| `research/contracts.py`、`cost_model.py` | `research/`：必要数值契约与成本逻辑 |
| `walk_forward_factory.py`、`overfit_guard.py`、`result_ranker.py` | `validation/`：时间窗口、启发式诊断与排序 |

新增统一 API、CLI、独立轻量数据类型、合成演示、文档、测试与构建配置。公式中的 permission/mode/gate 表示离线规则结果，不会授予任何外部账户权限。

## 已修正的边界

- 截止时间以后或尚未收盘的日线不进入个股研判。使用真实交易日及半日收盘时间。
- 相似状态统计的波动阈值只在训练段拟合；训练标签跨入测试段时排除。恒定价格的零波动阈值保持零，避免输出 Infinity。
- 评分中的真实零值不再被 `or` 默认值替换为中性分；增加有限值和配置校验，稳定计算对数收益。
- 日内数据要求单一股票、时间递增、合法 OHLC 和报价；增量相对成交量与批量版本使用同样的中位数基准。
- 在线模型与成本计算拒绝非有限输入，避免 NaN 借券费被当成零成本。
- 融合权重遵守单因子上限；不可行的上限配置显式拒绝，避免重新归一化后突破上限。
- 时间前推窗口实际应用训练／验证／测试长度和步长，单窗口内三段不重叠。不同滚动窗口仍可能复用观察；调用方需要按标签期限设置 purge/embargo。
- 验证与排序拒绝缺失或非有限证据；零回撤按真实零处理。
- 缺少必要数据时保留阻断原因；所有 proxy 统计保留启发式标签。

这些修改只应用于本公开副本。这里没有验证整套策略的投资收益。原逻辑中的固定阈值、价格代理、成本模型与分数仍需用户自行研究和独立验证。

## 依赖和数据

运行依赖通过包管理器安装，没有复制第三方依赖源码。演示行情由 `demo.py` 确定性生成，所有测试使用合成输入。发布包不含私有 Git 历史、下载的真实行情、账户配置、服务密钥或实盘执行适配器。
