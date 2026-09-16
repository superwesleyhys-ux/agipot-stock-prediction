# Optional framework integrations

The `agipot_stock_prediction.integrations` package connects supplied research
inputs to actual framework APIs. Importing it does not install anything or
start a model, server, browser, container, or data download. SDK imports happen
inside calls. A missing SDK raises `IntegrationUnavailable`; malformed input or
failed validation raises an error or returns an explicitly unsuccessful report.
A dependency being importable is never reported as a successful evaluation.

All examples below use caller-owned or synthetic data. None fetch market data.
The local tracking and report adapters disable supported SDK telemetry before
SDK import, setting `DO_NOT_TRACK=1`, `GX_ANALYTICS_ENABLED=False`, and
`MLFLOW_DISABLE_TELEMETRY=true` for the process. This does not turn arbitrary
user callbacks into a network sandbox. [GX analytics settings](https://docs.greatexpectations.io/docs/core/configure_project_settings/toggle_analytics_events/),
[MLflow telemetry settings](https://mlflow.org/docs/latest/community/usage-tracking/).

## Adapter map

| Catalog entry | Module / function | Actual operation and boundary |
|---|---|---|
| 18 Pandera | `data_quality.dataframe_schema`, `validate_frame` | Builds and runs `DataFrameSchema`, required/type/range/timezone/key checks; does not establish historical visibility. [API](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html) |
| 19 Great Expectations | `data_quality.validate_with_gx` | Ephemeral pandas batch, explicit expectations, per-check JSON results and combined success. No GX Cloud. [API](https://docs.greatexpectations.io/docs/core/connect_to_data/dataframes/) |
| 21 MLflow | `tracking.record_mlflow_run` | Local SQLite tracking client, local artifact directory, manifest artifact, metrics, final status. No tracking server or remote URI. [API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html) |
| 22 Evidently | `drift.evidently_report` | `Report([DataDriftPreset()]).run`, dictionary results and optional HTML. Drift is not accuracy. [API](https://docs.evidentlyai.com/docs/library/report) |
| 23 sklearn | `research.time_series_splits` | `TimeSeriesSplit` produces real row index partitions and requested `gap`; not label-interval purging. [API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) |
| 24 sktime | `research.evaluate_naive_forecast` | Expanding-window `evaluate` of `NaiveForecaster(strategy='last')`, refit each fold and MAE. Evaluates a baseline, not the AGIPOT predictor. [API](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.model_evaluation.evaluate.html) |
| 25 Qlib | `research.import_qlib_predictions`, `qlib_risk_analysis` | Local CSV/DataFrame prediction normalization plus actual `qlib.contrib.evaluate.risk_analysis` on supplied decimal returns. No provider initialization, market download, pickle, or order execution. [Source/API](https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py) |
| 26 Optuna | `research.tune_train_validation` | Seeded in-memory study, train/validation copies per objective invocation, retained COMPLETE/FAIL trial records; holdout suffix is not passed to the objective. [API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html) |
| 27 River | `river.make_adaptive_edge_regressor` | Constructs an actual River `base.Regressor` backed by `AdaptiveEdgeModel`; `predict_one`, `learn_one`, and clone work. Labels and predictions are in basis points. [API](https://riverml.xyz/latest/api/base/Regressor/) |
| 28 arch | `research.bootstrap_mean_interval` | Seeded `StationaryBootstrap.conf_int` for mean decimal returns, explicit block size/repetitions/confidence. Neither PBO nor DSR. [API](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html) |
| 39 Schemathesis | `optional.openapi_schema`, `smoke_openapi` | Loads supplied OpenAPI documents; explicit GET/HEAD case call and response validation for a caller-selected service. Document-external `$ref` is refused. [API](https://schemathesis.readthedocs.io/en/stable/reference/python/) |
| 40 Playwright | `optional.smoke_browser` | Installed Chromium opens an existing local HTML report or explicitly selected URL; checks page errors and requested visible text. Never installs browser binaries. [API](https://playwright.dev/python/docs/library) |
| 41 Testcontainers | `optional.smoke_container` | Uses a local Docker daemon and preloaded image; starts caller-selected command, checks its exit or an in-container service command, then removes container. No image pull, external networking, or Ryuk download. [API](https://testcontainers-python.readthedocs.io/en/latest/core/README.html) |
| 42 Promptfoo | `promptfoo.call_api` | Real Python-provider contract invoking shared `EvidenceTraceHarness`, deterministic original-text annotation checks; no LLM or paid API. [Provider contract](https://www.promptfoo.dev/docs/providers/python/) |

Other catalog entries such as DVC, Nox, pytest, CI and developer agents are
configured in their dedicated repository files rather than imported here.

## Canonical Pandera inputs

These optional research contracts are intentionally stricter than the base
analysis API's permissive raw inputs. Convert provider data explicitly; calling
`validate_frame` does not fill missing OHLC observations or publication times.
All four kinds require a nonempty DataFrame, nonempty `symbol`, and a
non-null timezone-aware `available_at`.

| `kind` | Additional required fields | Unique key |
|---|---|---|
| `daily` | `date` ISO calendar date, positive `open/high/low/close`, nonnegative `volume` | `symbol, date` |
| `minute` | timezone-aware `timestamp`, full OHLCV as above | `symbol, timestamp` |
| `tick` | timezone-aware `timestamp`, nonnegative integer `seq`, positive `price`, nonnegative `shares` | `symbol, timestamp, seq` |
| `fundamental` | `period_end` ISO date, nonempty `metric/unit`, finite `value`, timezone-aware `published_at`; optional aware `revision_at` | `symbol, period_end, metric, available_at` |

Unknown columns are retained. Fundamentals use long form (one metric per row);
negative fundamental values can be valid. Schema timezone checks do not imply
that an observation was visible at a requested `as_of`: run PointInTimeHarness
for that decision. GX is an independent batch report for required fields,
non-null values, numeric ranges and uniqueness; Pandera additionally handles
OHLC consistency and the declared timestamp/type contracts.

```python
import pandas as pd
from agipot_stock_prediction.integrations.data_quality import validate_frame

frame = pd.DataFrame([{
    'symbol': 'SYNTH', 'date': '2025-01-02',
    'open': 100, 'high': 102, 'low': 99, 'close': 101, 'volume': 200,
    'available_at': '2025-01-02T21:01:00Z',
}])
validated = validate_frame(frame, kind='daily')
```

## Train, validate, hold out

```python
from agipot_stock_prediction.integrations.research import tune_train_validation

def objective(trial, train, validation):
    scale = trial.suggest_float('scale', 0.5, 1.5)
    prediction = sum(train) / len(train) * scale
    return sum(abs(value - prediction) for value in validation) / len(validation)

study = tune_train_validation(
    [1., 2., 3., 4., 5., 6.], train_end=3, validation_end=5,
    objective=objective, n_trials=3, seed=7,
)
assert study['final_holdout_evaluated'] is False
```

The objective receives only the first two intervals, with fresh copies for each
trial. Python callbacks can still close over other variables; callers must not
capture holdout data. `n_jobs=1` avoids concurrent mutations, and failed numeric
trials remain in the returned records. This function does not compute a final
holdout metric. A separate frozen model evaluation must do that once.

`make_adaptive_edge_regressor` similarly does not decide when an online label is
mature: its `learn_one(x, y)` is an explicit update. Use the dedicated delayed
label replay harness to release labels after their availability timestamps and
to evaluate stored predictions before updating. Returns passed to Qlib/arch
are decimals (`0.01 = 1%`); River edge labels are basis points (`100 = 1%`).
Qlib's reported risk metrics follow its arithmetic sum accumulation convention,
with an explicit default 252 periods/year; they are not compounded CAGR.

## Local experiment and drift artifacts

```python
from agipot_stock_prediction.integrations.tracking import record_mlflow_run

result = record_mlflow_run(
    {'trial_id': 'synthetic-1', 'code_commit': 'fixture', 'data_fingerprint': 'example'},
    {'mae': 0.01}, output_dir='.agipot/mlflow', status='FINISHED',
)
```

`status` can also be `FAILED` or `KILLED`. The adapter saves the supplied manifest
as `experiment_evidence.json`. It does not invent a complete fingerprint or
claim all trials were recorded. Supply the registered evidence from
ExperimentEvidenceHarness; its append-only local log remains the completeness
check. A failing tracking callback must remain an integration error alongside
the underlying experimental outcome.

For offline drift, pass reference/current DataFrames with identical column
order to `evidently_report(reference, current, output_html='reports/drift.html')`.
No data is uploaded. The saved HTML may be opened with the browser smoke
adapter in local-report mode, which blocks HTTP(S) subrequests.

## Conditional service and browser checks

These calls are ready to use when their prerequisites exist. The base package
still has no HTTP application, production frontend, or required Docker service.

```python
from agipot_stock_prediction.integrations.optional import smoke_browser

# Requires an existing file and an already installed Playwright Chromium.
result = smoke_browser(html_path='reports/drift.html', expected_text='Data Drift')
```

URL mode requires `url='http://127.0.0.1:8000'` and `allow_network=True`; only that
origin is permitted. Schemathesis similarly requires an explicit `base_url`,
operation path, and `allow_network=True`; its smoke helper is one concrete case,
not a generated full API campaign. Use the returned schema's pytest
`parametrize()` API for a dedicated generated suite against an authorized test
service. External OpenAPI references must be bundled into the document first.

`smoke_container('your-local-image:tag', command=[...], allow_start=True)` checks
a local daemon and local image before creating a container. Pass
`check_command=[...]` to check an already-running service inside it. No ports,
volumes or credentials are injected. Image pulls are explicitly blocked, and
networking is disabled. The helper temporarily disables Testcontainers' Ryuk
helper and restores that setting in `finally`; Python context cleanup removes
the container, but abrupt process termination cannot provide Ryuk's cleanup
assurance. A missing/stopped daemon or absent image is `IntegrationUnavailable`.

## Promptfoo original-evidence regression

Use `file://.../integrations/promptfoo.py` as the Python provider and pass a JSON
prompt containing `claims`, `sources`, `entities`, `evidence`, and optional
`summary`. Dataclass field names match `validation.evidence_trace`. Timestamps
must be ISO strings with explicit offsets. `fixture_prompt()` supplies the
shared synthetic example, including a supported background claim, a refuted
causal claim, and an unknown same-name entity relation.

```python
import json
from agipot_stock_prediction.integrations.promptfoo import call_api, fixture_prompt

response = call_api(fixture_prompt(), {}, {})
result = json.loads(response['output'])
assert result['content_verdict'] == 'MIXED_SUPPORTED_AND_REFUTED'
assert result['automatic_truth_inference'] is False
```

Summary text never changes the evidence verdict. Support/refutation labels are
supplied annotations that require independent ground truth; this is an evidence
trace regression adapter, not automatic fact checking. Malformed prompts return
`error` without a fabricated `output`. The Python provider does not fetch the
Promptfoo CLI. The repository pins Promptfoo **0.122.1** in `package.json` and
`package-lock.json`; its CLI requires Node **22.22.0 or newer**.

From the repository root, after installing the Python package in an environment:

```bash
npm ci --omit=optional --no-audit --no-fund
PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE_CHECK=1 \
  PROMPTFOO_PYTHON=.venv/bin/python PYTHONPATH=src npm run test:evidence
```

`config/harness/promptfooconfig.yaml` runs four fixed synthetic cases: mixed
support/refutation, the same original evidence with a misleading summary,
missing evidence, and missing evidence with a confident summary. Alongside
independently specified atomic verdicts, each case compares the entire judgment
to a reviewed snapshot after removing only `display_summary`. Snapshots are
regression fixtures; update them only after reviewing the evidence contract.
Reports are written to the ignored `reports/promptfoo-results.json` path. The
configuration contains only the local Python provider and JavaScript/JSON
assertions; no model grader or paid API is configured.

The pinned version was selected after exercising both installation and audit.
On 2026-09-16, Promptfoo 0.123.0 with a full install produced three high-severity
audit entries along its optional `@openai/codex-security` / `extract-zip` chain
([advisory 1](https://github.com/advisories/GHSA-jmr9-qjv8-65gv),
[advisory 2](https://github.com/advisories/GHSA-7pqw-9j4j-h8q3)). Its
`npm audit --omit=optional` reported zero findings, but `npm ci --omit=optional`
also omitted the required platform SQLite binding and the CLI could not start.
The final 0.122.1 pin supports an ordinary `npm ci`; its complete dependency
audit reported **zero vulnerabilities**, and all four evidence cases passed.
Audit results describe that day's advisory database and should be rerun when
updating dependencies. No `npm audit fix --force` or platform-specific workaround
is required.

## Local development-agent prerequisites

The pinned agent manifest lives in `config/harness/agents.json`. Install project
skills, isolated executors, and the local gstack tools explicitly:

```bash
python scripts/bootstrap_agents.py
python scripts/bootstrap_agents.py --executors
python scripts/bootstrap_agents.py --build-browser
```

The browser build requires Bun and a native compiler on `PATH`. The explicit
flag runs `bun install --frozen-lockfile` and `bash scripts/build.sh` inside the
manifest's pinned `.harness/tools/gstack` checkout. It does not run gstack's
`setup` script. With Bun **1.3.10**, revision
`a6b3a57512ca6d5c6aa5b68f74f736195021f96e` built successfully through this entry
point: browser CLI and Node server, design/PDF tools, global-discovery binary,
and CSO binaries were produced inside the ignored checkout. The receipt is
`.harness/bootstrap-receipt.json`. Building does not invoke a model or log in.

The isolated **Aider 0.86.2**, **mini-SWE-agent 2.4.6**, and **RD-Agent 0.8.0**
executables returned successfully from `--help`. **OpenHands SDK 1.48.0** and
**openhands-tools 1.48.0** imported successfully; `LLM`, `Agent`, `Conversation`,
`Tool`, `FileEditorTool`, and `TerminalTool` expose the interfaces used by
`scripts/openhands_task.py`. These checks establish CLI/SDK readiness; actual
model execution still needs a configured backend and was not performed.

RD-Agent's upstream requirement leaves `pydantic-ai-slim` unbounded. Version
2.31.1 removed its imported `MCPServerStreamableHTTP` name, and MCP 2.2.0 removed
`RequestContext` needed by the v1 adapter. The manifest therefore also pins
`pydantic-ai-slim[mcp,openai,prefect]==1.107.5` and `mcp==1.30.0`. This pair
passed the actual CLI import and `uv pip check`; the installer reads these
constraints directly. See the upstream
[RD-Agent issue](https://github.com/microsoft/RD-Agent/issues/1432) and
[Pydantic AI v2 migration map](https://pydantic.dev/docs/ai/overview/migration/).

## Verification and prerequisites

`tests/test_integrations.py` exercises actual optional APIs when installed and
uses `pytest.importorskip` for absent dependencies. Missing prerequisites are
reported as skipped, not passing. Tests create synthetic frames, a temporary
local MLflow store, an offline drift HTML report, and a loopback HTTP fixture;
they do not contact market-data vendors or paid model services.

A separate Python 3.12 environment was used during implementation with:
Pandera 0.33.1, Great Expectations 1.23.0, MLflow 3.16.0, Evidently 0.7.23,
scikit-learn 1.7.2, sktime 0.40.1, Optuna 5.0.0, River 0.23.0, arch 8.0.0,
pyqlib 0.9.7, Schemathesis 4.27.2, Playwright 1.63.0, and Testcontainers 4.15.0.
These versions document the exercised API surface, not a claim that every
optional package combination or supported Python version was tested.

After explicitly installing Chromium **153.0.8010.12** (Playwright build **1243**)
into a temporary browser directory, the complete integration file finished
**32 passed, 0 skipped** on Python **3.12.6**. This includes actual local HTML
browser execution. Qlib risk analysis, dataframe schemas, batch validation,
local tracking, drift reporting, time splits, naive evaluation, hyperparameter
trials, online-regressor methods, bootstrap intervals, the loopback HTTP case,
and the deterministic evidence provider ran with synthetic data.

Promptfoo **0.122.1** was separately executed through its real `npx --no-install
promptfoo eval` CLI with Node **24.13.0** / npm **11.6.2**: **4 passed, 0 failed,
0 errors**. All four cases used the Python provider above. The local Docker
daemon was not running, so no Testcontainers container execution is claimed;
the Python tests exercise its explicit prerequisite gate only.
