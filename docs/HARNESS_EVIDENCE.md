# 实验完整记录与原文证据 Harness

这两套 harness 分别解决“预登记的实验是否全部有去向”和“关系判断能否回到指定版本的原文”。它们不把软件测试通过等同于预测有效，也不把来源文本存在等同于主张为真。模块默认只使用本地文件和调用者输入。

## E：ExperimentEvidenceHarness

实现：[`research/experiment_runner.py`](../src/agipot_stock_prediction/research/experiment_runner.py)。

`ExperimentSpec` 要求记录所有实际使用的输入及版本、完整配置、代码版本及代码内容摘要、依赖锁摘要与实际依赖版本、标签定义、划分协议、随机种子及输入清单规则。每项 `inputs[name]` 必须包含 `version` 和 `content`。输入不能只覆盖日线而遗漏实际用到的分钟、ticks、财报、模型权重或外部特征。

`spec.fingerprint()` 对规范化 JSON 做 SHA-256：映射键排序，数组顺序保留，时间转 UTC；拒绝 NaN、Infinity、无时区 datetime、非字符串字典键和未定义的对象类型。代码提交号不能代表未提交修改，所以另外要求 `code_digest`；可对完整代码树的 `{相对路径: digest_file(文件路径)}` 映射再调用 `fingerprint`，不能只哈希版本标签。用 `digest_file(lock_path)` 记录真实锁文件摘要；`capture_dependency_versions(names)` 记录解释器及指定的已安装依赖版本。调用者应传全本次实际使用的依赖名。

```python
from random import Random
from tempfile import TemporaryDirectory
from agipot_stock_prediction.research.experiment_runner import (
    ExperimentRunner, demo_experiment_specs,
)

def worker(spec):
    rng = Random(spec["seed"])
    prices = spec["inputs"]["synthetic_prices"]["content"]
    return {
        "metrics": {"synthetic_statistic": sum(prices) / len(prices)},
        "seeded_draw": rng.random(),
        "sample_count": len(prices),
        "uncertainty_interval": "NOT_COMPUTED",
        "serial_dependence_adjustment": "NOT_COMPUTED",
    }

with TemporaryDirectory() as directory:
    trials = demo_experiment_specs()
    runner = ExperimentRunner(directory, trials)
    outcomes = [runner.run(trial.trial_id, worker) for trial in trials]
    assert all(item["status"] == "SUCCESS" for item in outcomes)
    print(runner.verify_integrity())
```

### 登记、失败、重试与产物

创建 runner 时先将整个 trial 清单写入 `plan.json`，再追加登记事件；已有清单不能原地增删。`run(trial_id, worker)` 只接受清单内、尚未开始的 trial。worker 获得预登记规范的独立快照，修改该字典不会修改计划。runner 不修改全局随机状态，worker 应显式使用登记的种子。

`events.jsonl` 记录 `trial_registered`、`trial_started`、`trial_succeeded`、`trial_failed`、`trial_aborted` 和 `trial_retry`。每个事件包含序号、前一事件哈希、自身哈希、UTC 时间、trial ID 和 attempt。文件追加后执行 `fsync`，使用 SQLite 事务锁串行化本地写入。SQLite 文件只负责进程间锁，审计正文仍是 JSONL。

普通 worker 异常记为 `FAILED`，并返回失败结果；`KeyboardInterrupt` / `SystemExit` 记为 `ABORTED` 后继续抛出。调用者必须读取 `run()` 的 `status`，不能把“函数已返回”解释成实验成功。`run(..., retry=True)` 仅可重试失败或中止的同一规范，并保留前次失败事件；更换数据或配置应登记新的 trial。`abort(trial_id, reason)` 可以记录预登记后的跳过或已停止进程的中断；它不会终止仍在执行的进程。

worker 可返回 `{"artifact_paths": {"model": "/path/to/model.bin"}, ...}`。runner 将这些文件复制到本实验目录，返回结果中的 `artifacts` 会记录相对路径和 SHA-256。`artifacts` 为 runner 保留字段。成功结果整体也有 `result_fingerprint`；完整性检查会重新核验已保存产物，原文件随后变化不影响已归档副本。

### 完整性检查的范围

`verify_integrity()` 默认要求每个预登记 trial 都有终态，校验计划摘要、哈希链、注册和状态转换、结果摘要及归档产物。试验仍在执行时可用 `require_terminal=False` 查看状态。它会拒绝以下情况：

- 一个登记的失败 trial 被整个移出日志，即使剩余日志已重新计算哈希。
- 已开始 trial 的失败或成功终态被删除。
- 事件正文变化、乱序、断链或最后一行不完整。
- 计划在创建后被直接替换，或产物缺失、变化。

日志证明的是相对于当前冻结计划的完整记账，不能阻止拥有全部文件写权限的人同时重写计划和日志，也不能自动发现绕过 runner 执行的试验。应在受控存储保存报告中的 `plan_fingerprint` 和 `head_hash`。后来可将它们传入 `expected_plan_fingerprint=`、`expected_head_hash=`，检测整体替换或回退到较早的合法日志前缀。

对于未来且没有被使用的数据，按 `input_scope` 中声明的规则生成规范：若规则是只记录该时点可见且实际使用的记录，增加不可见的未来行可以不改变指纹；若规则包含整个原始快照，新增行或版本变更应改变指纹。runner 不自行猜测或删除输入，也无法阻止 worker 访问未声明的外部状态。

报告保留 `formal_pbo=NOT_COMPUTED`、`formal_dsr=NOT_COMPUTED`、`statistical_validity=NOT_ESTABLISHED_BY_ACCOUNTING`。样本量、划分、重复观察处理、时间相关性处理和不确定性区间仍须由研究评估阶段提供。

### 可选跟踪回调

`ExperimentRunner(..., on_event=callback)` 在事件本地落盘后调用 `callback(event_dict)`。事件字典包含 `event_type`、`trial_id`、`attempt`、`payload`、`sequence`、`previous_hash`、`event_hash`、`timestamp`。`trial_registered.payload` 包含完整规范及规范指纹，成功 payload 包含结果和结果指纹。

此接口可接 MLflow 等外部跟踪器。回调异常另记 `integration_error`，本地 trial 结果仍保留；原事件可根据序号和哈希重放到恢复后的跟踪器。回调不应重入同一 runner。外部跟踪页面不能取代计划及追加日志核对。

## F：EvidenceTraceHarness

实现：[`validation/evidence_trace.py`](../src/agipot_stock_prediction/validation/evidence_trace.py)。

这是**证据契约与固定真值评测工具，不是自动真实性推理模型**。它不自动从新闻抽取关系，不自动判断句子蕴含，不用另一个模型的总结替代证据。实体消歧、原子关系、证据支持或反驳的注释、来源转载关系及时间适用范围，应来自人工真值或另外经过评估的抽取器。

| 对象 | 关键字段及语义 |
| --- | --- |
| `SourceDocument` | `source_id`、`version`、`text`、带时区 `published_at`、原始来源 `origin_id`；返回证据中另存原文 SHA-256 |
| `Entity` | 稳定的 `entity_id` 和展示 `label`；同名 label 不合并 ID |
| `AtomicClaim` | `claim_id`、`subject_id`、`predicate`、`object_id`、带时区 `as_of`；每项只表达一个原子关系 |
| `EvidenceSpan` | 指定来源 ID/版本、`start/end/quote`、同样的原子关系、`verdict`、适用时间；verdict 为 `supports/refutes/unknown` |

偏移按 Python Unicode 字符位置，区间是 `text[start:end]`，不是 UTF-8 字节偏移。引用必须与指定版本原文精确相等，关系的两个端点都必须是已登记的实体 ID，谓词及端点必须完全匹配主张。仅关键词共现不产生关系。

来源须在主张 `as_of` 时已经发布。时间适用范围为 `[valid_from, valid_until)`；未填写完整范围的证据不会默认永久有效。确实不随时间变化的关系可由注释者显式声明 `timeless=True`，并且不再填写区间。过时、未来或无法定位的证据会保留为 `rejected_evidence`，不会强行变为反证。

```python
from agipot_stock_prediction.validation.evidence_trace import (
    demo_evidence_fixture, evaluate_evidence,
)

result = evaluate_evidence(
    **demo_evidence_fixture(),
    summary="仅展示用途；即使摘要声称全部属实，也不能覆盖原文证据。",
)
assert {item["claim_id"]: item["verdict"] for item in result["claims"]} == {
    "background": "SUPPORTED",
    "false-cause": "REFUTED",
    "same-name-error": "UNKNOWN",
}
assert result["content_verdict"] == "MIXED_SUPPORTED_AND_REFUTED"
```

这个合成样本包含两个均展示为 Aurora、但稳定 ID 不同的主体。原文支持一家企业开设实验室，并明确否认“该实验室导致另一家企业盈利下滑”的因果关系。第一项为支持，第二项为反驳；同名另一家企业开设实验室的主张没有证据，保持未知。

### 真值与来源计数规则

- 原子主张有合格支持证据时为 `SUPPORTED`，有合格反驳证据时为 `REFUTED`，两者同时存在为 `CONFLICTED`，都没有则为 `UNKNOWN`。
- 一条内容中同时出现 `SUPPORTED` 和 `REFUTED` 主张，才标记 `MIXED_SUPPORTED_AND_REFUTED`，表达“半真半假”的主张级混合状态。它不代表恰好 50% 的比例。
- 真实背景加无证据的因果主张为 `SUPPORTED_WITH_UNKNOWN`；缺证不判假。单个原子主张的矛盾证据另标 `CONFLICTED_EVIDENCE`，不冒充已确定的真假混合。
- 同一 `origin_id` 或全文完全相同的转载折叠为同一来源组；仅使用各主张 `as_of` 时已经发布的来源构建分组，未来转载不能反向改变历史计数。保留全部定位链，但独立支持数不重复增加。未知的改写转载仍可能相关，因此计数不等于统计独立性证明。
- `summary` 只出现在 `display_summary`。修改摘要不会改变原子判断、证据筛选或整体标签。

固定真值测试覆盖原文版本/偏移、伪造引用、同名主体、关键词共现、转载、失效区间、未来来源、缺证未知、支持与反驳混合，以及摘要“洗白”不改变结果。接口可以接受人工标注结果，也可由 Promptfoo 回归适配器调用；模型输出仍需要与这些固定真值比较。

## 本地验证

```bash
PYTHONPATH=src python -m pytest tests/test_experiment_runner.py tests/test_evidence_trace.py -q
```

测试完全使用合成输入和临时目录，没有下载真实新闻或交易数据。源文件与测试通过只证明这些契约及确定性案例的实现行为，不证明自动事实核查准确率或市场预测能力。
