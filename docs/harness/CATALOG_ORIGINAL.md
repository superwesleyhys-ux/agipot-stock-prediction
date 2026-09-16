# agipot-stock-prediction：研发到测试的 Harness 选型与接入建议

核查日期：2026-09-16。仓库：[superwesleyhys-ux/agipot-stock-prediction](https://github.com/superwesleyhys-ux/agipot-stock-prediction)。代码基准：[609368b33a723e99bd679cab45b35b9f5dd62cbd](https://github.com/superwesleyhys-ux/agipot-stock-prediction/tree/609368b33a723e99bd679cab45b35b9f5dd62cbd)。

本清单按项目全生命周期覆盖 **42 组现成工具／框架，以及 6 个项目专用验证 harness**。它不是对开源生态全部项目的穷举。能力描述根据官方项目／文档核查；优先级和接入方式是结合当前代码的工程建议，未代表全部候选已经安装或验证兼容。

建议主组合：**Superpowers + pytest / Hypothesis / mutmut + Pandera + 自建时间验证 runner + DVC / MLflow + GitHub Actions**。已有 Codex 时，Aider、OpenHands、mini-swe-agent 属于替代执行器或专门实验选项，不需要全部叠加。

## 1. 这个仓库当前有什么

当前是 Python 3.11+ 的离线研究与评分工具包，包含公式、日线评分、日内特征、在线学习组件、CLI 和 Python API。`api.py` 是本地 Python 接口，仓库没有 HTTP 服务或 Web 前端。[项目说明](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/README.md)

本次在独立虚拟环境、Python **3.12.14** 下安装项目现有 `.[dev]` 后实测：

- `python -m pytest -q`：**122 passed in 1.97s**。
- `python -m ruff check .`：**All checks passed**。
- 检查后 Git 工作区无修改。本次没有重跑构建产物安装验证，也没有运行其他 Python 版本或新增框架的兼容性测试。

这些结果验证现有软件测试通过，不代表预测能力、无时间泄漏或统计可靠性已经得到证明。

| 已有基础 | 当前缺口与含义 |
|---|---|
| pytest、Ruff、build 已在开发依赖中 | Ruff 目前只启用 `E9/F63/F7/F82`；尚未配置 coverage、性质测试、变异测试、类型检查。见 [pyproject.toml](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/pyproject.toml)。 |
| 文档有 Python 3.11/3.12/3.13 的 CI 模板 | 当前没有 `.github/workflows`；[模板仍在 docs](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/docs/github-actions-tests.yml)，不能当成已启用 CI。 |
| walk-forward 日历窗口生成 | 尚未执行逐折训练、折内预处理、按标签时间 purge 或结果汇总。见 [walk_forward_factory.py](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/src/agipot_stock_prediction/validation/walk_forward_factory.py)。 |
| distillation 已有基本防泄漏措施 | `_current_regime_edge` 已有 70/30 时间 holdout、仅训练段拟合波动阈值、5/20 日跨边界标签 purge；仍不是完整多折实验框架。见 [distillation.py](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/src/agipot_stock_prediction/distillation.py)。 |
| 在线学习组件 `AdaptiveEdgeModel` | `learn` 没有成熟标签队列、样本身份或时间参数；调用者负责标签何时可用。见 [signal_library.py](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/src/agipot_stock_prediction/intraday/signal_library.py)。 |
| overfit guard 已明确标注证据边界 | 输出 `heuristic_only`，正式 PBO/DSR 未计算，OOS 未验证；`all_trials_recorded` 还是调用者断言。见 [overfit_guard.py](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/src/agipot_stock_prediction/validation/overfit_guard.py)。 |
| 日线／ticks 已参与部分数据哈希 | 当前 `data_hash/run_id` 未覆盖所有影响结果的 fundamentals、minute_rows、完整配置和代码版本；还不能作为完整实验身份。见上述 distillation 源码。 |

## 2. 先区分三种东西

- **研发代理 harness／skill pack**：安排代理如何规划、改代码、调用工具、审查和验证，如 Superpowers、OpenHands SDK。
- **验证工具与研究基础设施**：执行测试、数据校验、实验跟踪，如 pytest、Pandera、MLflow。它们通常被 harness 调用。
- **项目专用验证 harness**：把输入、时间、执行顺序、判定规则和证据串起来。现成工具不能自动替项目定义“当时是否可见”“标签是否成熟”。

优先级：**P0 = 先补；P1 = 随研究评估接入；P2 = 特定需求出现再选**。同一行的替代版本算一组。每个工具名称链接均为官方来源。

## 3. 研发规划、编码与审查：7 组

| # | 工具／官方来源 | 用在本仓库哪里 | 优先级与接入判断 |
|---|---|---|---|
| 1 | [Superpowers](https://github.com/obra/superpowers) | 需求澄清、实施计划、测试驱动修改、根因调试、代码审查、完成前验证 | P0；作为主要研发流程。官方支持 Codex App／CLI；属于技能包，不是 pytest 替代品。 |
| 2 | [GitHub Spec Kit](https://github.com/github/spec-kit) | 大功能先写输入输出、数学假设、失败状态、验收要求，再拆任务 | P1；[官方集成列表](https://github.github.com/spec-kit/reference/integrations.html)包含 Codex。和主流程共享一份规格，避免维护两套需求。 |
| 3 | [gstack](https://github.com/garrytan/gstack) | 选用工程计划审查、代码审查、调查调试、发布文档模块 | P1；当前官方支持 Codex CLI host。浏览器 QA 部分暂时无目标；与 Superpowers 重叠的流程择一使用。 |
| 4 | [Aider](https://github.com/Aider-AI/aider) | 集中修复／重构；通过 [lint/test 命令](https://aider.chat/docs/usage/lint-test.html)驱动修改反馈 | P2；独立终端代理、替代执行器。已用 Codex 不必同时再让它修改相同文件。 |
| 5 | [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) | 将任务分配、独立工作区、代码修改、测试和执行日志编程化 | P2；中高接入成本，适合长期研发自动化服务。核心运行时看 SDK；独立模型后端需另行配置。 |
| 6 | [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)／[SWE-agent](https://github.com/SWE-agent/SWE-agent) | 对固定 bug 集比较代理修复率、耗时、轨迹和消耗 | P2；多数小型实验先选 mini；研究更复杂工具界面／历史策略再看完整版。 |
| 7 | [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) | 离线研究假设、实现实验、执行反馈、模型研发迭代 | P2；先冻结数据、评估协议和实验预算，再考虑自动化研究循环。不能代替独立最终测试集。 |

## 4. 本地工程质量与软件测试：10 组

| # | 工具／官方来源 | 具体接入位置与收益 | 优先级 |
|---|---|---|---|
| 8 | [uv](https://docs.astral.sh/uv/) | 统一开发环境与锁文件，固定研究环境依赖；可复现安装 | P0；锁定测试环境，同时保留库对外声明的兼容版本范围。 |
| 9 | [Ruff](https://docs.astral.sh/ruff/) | 全包 lint；逐步扩展当前狭窄规则集，按需统一格式 | P0；已有，先增强配置；格式整理与公式行为修改分开审阅。 |
| 10 | [mypy](https://mypy.readthedocs.io/en/stable/) | 从 `contracts.py`、公共 API、时间／数值类型边界开始类型检查 | P1；渐进启用，不能用全局忽略消除实质错误。 |
| 11 | [pre-commit](https://pre-commit.com/) | 本地提交时运行轻量 lint、格式和文本检查 | P1；提升反馈速度，CI 仍运行对应检查。 |
| 12 | [pytest](https://docs.pytest.org/en/stable/) | 保留当前 6 个测试文件；补公共契约、异常输入、CLI 与集成测试 | P0；已有 122 测试，本次全部通过。 |
| 13 | [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) | 对 `formulas/`、`scoring/`、`validation/` 查看行与分支覆盖 | P0；先测基线，再定新增／关键路径门槛；覆盖率不等于公式正确。 |
| 14 | [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) | 生成 NaN、Infinity、零值、缺失、重复时间、DST、窗口端点等；做性质与状态机测试 | P0；重点补手写样例没覆盖的组合，保留已有回归测试。 |
| 15 | [mutmut](https://mutmut.readthedocs.io/en/latest/) | 对 `< / <=`、正负号、缺失分支、成本扣除等关键逻辑做变异测试 | P0，定向运行；先测纯函数与关键判定，审阅存活变异；按官方平台要求配置运行环境。 |
| 16 | [pytest-regressions](https://pytest-regressions.readthedocs.io/en/latest/) | 固定 CLI JSON、DataFrame 与数值输出；保护字段、状态和单位 | P1；基线变更必须解释；快照不能替代独立数学预期。 |
| 17 | [Nox](https://nox.thea.codes/en/stable/) | 同一入口编排 lint、tests、构建、wheel 安装后测试及 Python 版本矩阵 | P0；复用现有命令，减少本地与 CI 差异。 |

## 5. 数据质量、可复现性与实验记录：5 组

| # | 工具／官方来源 | 具体接入方式 | 优先级与边界 |
|---|---|---|---|
| 18 | [Pandera](https://pandera.readthedocs.io/en/stable/) | 校验日线、分钟、ticks、财报 DataFrame 的字段、类型、唯一性、范围和时间列 | P0；适合当前 pandas 技术栈。历史时点可见性仍须项目逻辑定义。 |
| 19 | [Great Expectations / GX Core](https://docs.greatexpectations.io/docs/core/introduction/) | 数据批次验收与可读质量报告 | P2；数据源较多时作为更重的选项；初期可只用 Pandera。 |
| 20 | [DVC](https://doc.dvc.org/) | 数据快照、数据处理阶段和版本关联；保存可追踪的输入身份 | P1；与 Git 的代码版本配套，数据保留在合适的数据存储中。 |
| 21 | [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) | 每次实验记录数据版本、代码提交、参数、种子、划分、指标、产物与失败状态 | P1；由统一 runner 自动登记，避免仅记录成功试验。 |
| 22 | [Evidently](https://docs.evidentlyai.com/introduction) | 数据分布／特征漂移和离线评估报告；与历史基线比较 | P2；漂移检测不等于预测准确性检验，也不能证明无时间泄漏。 |

## 6. 时间序列、离线历史评估与统计验证：6 组

| # | 工具／官方来源 | 适用内容 | 接入判断 |
|---|---|---|---|
| 23 | [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | 标准时间序列拆分及 `gap`，作为基线与对照 | P1；不能把固定样本数 gap 当成按标签区间执行的 purge／embargo。当前窗口生成器可以保留。 |
| 24 | [sktime](https://www.sktime.net/) | 标准时间序列预测接口、切分与预测器评估 | P2；引入标准预测模型比较时适用；当前评分函数需适配。 |
| 25 | [Microsoft Qlib](https://github.com/microsoft/qlib) | 更完整的离线数据、模型和研究评估流程 | P2；独立研究框架，接入成本高，需要统一数据／模型／指标契约；当前不必迁移整个项目。 |
| 26 | [Optuna](https://optuna.readthedocs.io/en/stable/) | 配置与模型超参数实验；统一保存试验记录 | P1，有调参需求时启用；限定在训练／验证阶段，最终 holdout 不参与选择。 |
| 27 | [River progressive validation](https://riverml.xyz/latest/api/evaluate/progressive-val-score/) | 在线模型先预测、标签可用后再更新；支持 `moment` 和 `delay` | P1；为 `AdaptiveEdgeModel` 加适配层；默认没有延迟，必须显式提供实际标签成熟规则。 |
| 28 | [arch 时间序列 bootstrap](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html) | 用 stationary／block bootstrap 评估相关时间序列指标的不确定性 | P1；块长度和统计量需明确；不是正式 PBO/DSR 的自动替代品。 |

研究验收要分开记录：公式实现是否符合定义、样本外预测误差／方向或排名指标、跨时期稳定性、实验选择过程。现有 score 不能直接当作已校准概率；现有 proxy 也不能改个名称就变成正式统计检验。

## 7. 性能、内存与长期回归：3 组

| # | 工具／官方来源 | 对应代码／用途 | 优先级 |
|---|---|---|---|
| 29 | [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) | `_current_regime_edge`、日内特征生成、评分批处理的函数级基准 | P1；先固定输入规模和环境，避免将共享 CI 机器抖动当成性能退化。 |
| 30 | [ASV](https://asv.readthedocs.io/en/stable/) | 跟踪多个提交的速度、内存和自定义性能指标 | P2；需要长期趋势时采用，不必一开始与 pytest-benchmark 重复建设。 |
| 31 | [Memray](https://bloomberg.github.io/memray/) | 批量计算和 streaming 状态的内存分配分析 | P2；出现具体内存风险时定位；按平台支持安排运行环境。 |

## 8. 安全扫描、CI 与包交付：7 组

| # | 工具／官方来源 | 用途 | 优先级 |
|---|---|---|---|
| 32 | [Bandit](https://bandit.readthedocs.io/en/latest/) | 检查 Python 常见不安全代码模式 | P1；静态扫描结果需人工判断上下文。 |
| 33 | [pip-audit](https://github.com/pypa/pip-audit) | 扫描测试／运行依赖的已知漏洞 | P0；结合锁定环境和更新流程使用。 |
| 34 | [Gitleaks](https://github.com/gitleaks/gitleaks) | 检查代码与提交中的密钥泄漏 | P0；本地和 CI 都可运行，报告避免再次暴露密钥。 |
| 35 | [GitHub CodeQL](https://docs.github.com/en/code-security/concepts/code-scanning/codeql/codeql-code-scanning) | GitHub 上的代码安全分析 | P1；较深入扫描，按仓库适用配置启用。 |
| 36 | [GitHub Actions](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) | PR 与提交触发 lint、测试、版本矩阵、构建和报告产物 | P0；把现有 docs 模板正式落到 `.github/workflows/` 才会执行。 |
| 37 | [PyPA build](https://build.pypa.io/en/stable/) | 构建 wheel／sdist；在独立环境验证已安装 wheel | P0；已有开发依赖，建议纳入 CI；确保不误从源码目录导入。 |
| 38 | [Twine](https://twine.readthedocs.io/en/stable/) | 使用 `twine check` 检查分发包元数据／长描述 | P1；配合 build 做交付校验。此处仅建议检查，不涉及自动上传包。 |

## 9. 架构扩展后才启用：4 组

| # | 工具／官方来源 | 启用条件 | 当前判断 |
|---|---|---|---|
| 39 | [Schemathesis](https://schemathesis.readthedocs.io/en/stable/) | 将 Python API 封装为 OpenAPI／GraphQL 服务后，生成接口契约测试 | P2；当前本地 `api.py` 用 pytest 即可。 |
| 40 | [Playwright](https://playwright.dev/python/) | 有 Web 界面后，测试用户流程和页面交互 | P2；当前没有前端，不列为首批。 |
| 41 | [Testcontainers Python](https://testcontainers-python.readthedocs.io/en/latest/) | 引入数据库／缓存／队列等服务后，做真实依赖的集成测试 | P2；需容器环境；纯内存函数阶段收益有限。 |
| 42 | [Promptfoo](https://www.promptfoo.dev/docs/intro/) | 接入 LLM 的关键词／关系提取、证据判断或新闻解析后，跑固定样本和提示词回归 | P2；可承接你前面提的“关键词—关联—溯源”评测，但须独立设计证据真值；不能由另一个模型的评价代替来源证据。 |

## 10. 最值得自建的 6 个 Harness

下面名称是建议的新模块，不是声称仓库已经存在这些实现。相对路径仅表示拟新增位置。

### A. PointInTimeHarness：验证历史时点可见性

建议位置：`src/agipot_stock_prediction/validation/point_in_time.py`；测试放 `tests/test_point_in_time.py`。

输入至少记录 `event_time`、`available_at`、`as_of`、数据版本；财报补 `published_at/revision_at`；模型补 `trained_through/model_available_at`；日内观察补完成时间。规则按数据源与接口契约明确定义。

关键验收：添加未来披露的财报，历史 `as_of` 输出不变；修订数据只有在修订可用后才被读取；尚未结束的分钟观察不能参与该时点计算；模型必须在推断时点前已完成训练并可用，所有训练标签和预处理输入也必须满足当时训练截止时点的可用性要求。不能仅凭 `trained_through <= as_of` 判定无泄漏。Pandera 检查结构，本 harness 检查时间语义。

保留现有接口差异：`scoring` 只取 `as_of` 日期之前日线，趋势部分还跳过最近 5 个完整记录；distillation 使用已完成交易日判定。不能为“统一测试”强行改成相同截止逻辑。见 [scoring/api.py](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/src/agipot_stock_prediction/scoring/api.py) 与 [calendar.py](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/src/agipot_stock_prediction/calendar.py)。

### B. WalkForwardHarness：把日历窗口变成可审计实验

建议位置：`src/agipot_stock_prediction/validation/walk_forward_runner.py`；复用现有 `build_walk_forward_windows`。

每折重新创建模型及预处理器，只在该折训练数据拟合；标签可用时间必须满足训练截止规则；根据标签信息区间处理跨边界重叠。需要间隔时明确 embargo 规则，并与普通样本数 gap 区分。验证阶段用于选择，最后冻结一段最终 holdout。

关键验收：篡改 holdout 值不改变已拟合训练参数；训练样本的标签全部已成熟；窗口端点包含半日、假日、闰日测试；跨折重复预测先规定去重／聚合规则，再计算整体指标，不能把重复观察当独立样本。

### C. DelayedLabelReplayHarness：在线模型按事件时间重放

建议位置：`src/agipot_stock_prediction/validation/delayed_label_replay.py`；为 `AdaptiveEdgeModel` 增加事件适配器。

事件顺序是：可用特征到达→保存预测→等待对应标签成熟→用当时保存的预测评估→学习。显式记录 `sample_id/predicted_at/label_available_at/learned_at`，定义乱序、重复与缺失事件策略。

关键验收：标签成熟前不学习；同一样本不重复更新；重复重放得到相同结果；不能用标签成熟后的模型重新计算过去预测。River 可提供渐进评估机制，项目仍需自己的时间和幂等契约。

### D. FormulaContractHarness：测试公式、单位和流式一致性

建议位置：`tests/test_formula_properties.py`、`tests/test_streaming_properties.py`。

用手算小样本或独立参考实现作 oracle；通过 Hypothesis 生成边界输入；对关键比较、符号和空值判断做 mutmut。明确浮点容差、单位、空值含义、适用域内的单调性和边界。

关键验收：零值与缺失不能混同；固定其他输入时，增加非负成本不应提高对应净指标；同一有序输入的 batch 与 streaming 结果在约定容差内一致；关键错误变异会使测试失败。现有 batch／streaming 和未来价格回归测试继续保留。

### E. ExperimentEvidenceHarness：记录完整实验，而不只记录最佳结果

建议位置：`src/agipot_stock_prediction/research/experiment_runner.py`；由 DVC／MLflow 配套。

完整实验清单包含代码提交、依赖锁摘要、所有实际使用的输入版本、配置、随机种子、标签定义、分割协议和模型产物。新建完整 fingerprint；不要直接将当前不完整的 `data_hash/run_id` 解释为完整身份。

所有 trial 先登记再执行，记录成功、失败、中止和重试；用预登记任务清单、唯一 ID 和追加式事件日志核对完整性。单靠 MLflow 页面或 `all_trials_recorded=True` 不能证明没有遗漏，必要时限制删除权限并保留外部日志。

关键验收：任何影响结果的已使用输入／配置／代码变化都会改变完整身份；未来且未使用的记录遵循清单规则；缺失一个已登记失败 trial 时完整性检查失败。统计报告保留样本量、划分、相关性处理及区间估计；正式 PBO／DSR 未实现时继续标明未计算。

### F. EvidenceTraceHarness：关键词、关联与来源证据的逐条评测

这是接续你前面“不要按 summary 判断洗白信息”的需求；仅在本项目引入新闻／证据信号时接入。建议放独立适配模块与固定评测集，不把当前打开的 patch 当作本仓库已经有的代码。

评测输入应是原文及来源，先提取实体／事件／关键词，再进行实体消歧和关系核验，最后逐个原子主张回指原文片段、时间和来源。摘要仅用于展示，真实性判定输入为原文定位的关系及原始证据，不以生成摘要替代它们。

关键验收包括：同名主体不误关联；同一来源的多次转载不算独立佐证；真实背景加虚假因果应分别标注；证据缺失输出未知；过时证据不能支持新时点主张。保存原文版本、定位与证据链；关键词共现本身不构成事实关系。

“半真半假”定义为同一条内容含有证据支持的主张以及被证据反驳的主张／错误关联；无证据的部分保留未知，不强制判假。该标签表达主张级混合状态，不预设 50% 的真假比例，也不承诺能保证全部识别。此 harness 可用 pytest 的确定性断言和 Promptfoo 的模型回归共同执行。

## 11. 建议落地顺序与验收

| 批次 | 交付内容 | 验收依据 |
|---|---|---|
| 第一批：工程基线 | 固定环境；启用 Actions／Nox；保留 122 测试；补 coverage、定向 Hypothesis／mutmut；build + wheel 安装验证 | 现有测试通过；关键错误变异被发现；独立环境从 wheel 导入；CI 中的 Python 矩阵实际运行。 |
| 第二批：时间与数据 | Pandera + PointInTimeHarness + 完整输入／模型时间契约 | 未来财报、未完成观察、未来模型权重无法影响历史时点输出。 |
| 第三批：研究评估 | WalkForwardHarness + DelayedLabelReplayHarness + DVC／MLflow 完整试验记录 | 折内训练无泄漏；成熟标签才学习；失败实验可追踪；最终 holdout 不参与调参。 |
| 第四批：扩展能力 | 按需求补 Optuna、漂移／性能监测、研发代理对照实验或 EvidenceTraceHarness | 每项扩展对应实际需求、固定评测集和可审阅产物。 |

首批不需要迁移到大型研究平台，也不需要同时安装多套编程代理。采用一个研发主流程，让它调用明确的测试命令，并要求每次结果附带代码版本、数据版本、测试输出及未验证项。

## 12. 使用与证据边界

本次交付的是仓库审计与选型建议，未将候选工具全部接入。软件基线实测仅为上述 Python 3.12.14 的 pytest 与当前 Ruff 规则；其他能力来自官方文档核查，优先级来自工程分析。项目的新配置／新模块／CI 路径均为建议，未声称已经提交。

构建与统计验收应遵循仓库 [CONTRIBUTING.md](https://github.com/superwesleyhys-ux/agipot-stock-prediction/blob/609368b33a723e99bd679cab45b35b9f5dd62cbd/CONTRIBUTING.md) 中对确定性测试、合成数据和证据区分的要求。新 harness 应保留当前 `research_only`／未验证状态语义，直到对应证据确实存在。
