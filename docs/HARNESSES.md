# Harness integration

The supplied catalog is implemented as 42 explicit tool integrations and six
project-owned research harnesses. An entry in the catalog is not a test result.
`doctor` reports dependency availability; execution receipts and the verification
record distinguish passed checks, findings, and unmet prerequisites.

## Quick start

```sh
uv sync --locked --extra harness --extra data
uv run agipot-harness catalog
uv run agipot-harness doctor
uv run agipot-harness demo
uv run agipot-harness run properties
uv run nox
```

`demo` runs A–F on synthetic observations and writes a separate run
folder under `reports/harness-demo/` without overwriting previous demo runs. The
local files remain writable; hashes and external checkpoints provide integrity
checks, not write protection. It includes expected failed/aborted
experiments, validates their retained evidence, and never asserts profitability.
Named profiles save command output, exit status, source digest and lock digest
under `reports/harness/`. A failed command makes the profile fail. Missing tools
remain explicit. These files are local artifacts, excluded from Git.

## Six research harnesses

| ID | Harness | Actual contract |
|---|---|---|
| A | PointInTimeHarness | Select the version visible at a historical cutoff; reject model/training/label/preprocessor time leakage. |
| B | WalkForwardHarness | Fresh fit per fold, mature labels, interval purge/embargo, validation-only model selection and frozen holdout. |
| C | DelayedLabelReplayHarness | Predict first, preserve the original prediction, evaluate then learn when the label arrives. |
| D | FormulaContractHarness | Independent cost/unit oracle, ensemble weights, generated boundaries and batch/stream comparisons. |
| E | ExperimentRunner | Pre-register all trials; preserve success, failure, abort and retry evidence with content fingerprints and an append-only hash chain. |
| F | EvidenceTraceHarness | Source version and character spans, stable entity IDs, claim-level time validity and syndication deduplication. |

See [temporal APIs](HARNESS_TEMPORAL.md), [experiment and evidence APIs](HARNESS_EVIDENCE.md),
and [third-party adapter APIs](INTEGRATIONS.md). Annotated evidence is not an
automatic truth oracle. Local hash chains require an external checkpoint to
detect a privileged rewrite of the entire history. Generic callbacks must not
read undeclared future or holdout data. None of these checks prove market alpha.

## All 42 catalog entries

The last column identifies a checked-in execution path, or the explicitly linked
GitHub-hosted CodeQL run. Configuration alone does not mean an external service
ran. See [verification record](harness/VERIFICATION.md) for actual runs.

| # | Tool | Integration |
|---|---|---|
| 1 | [Superpowers](https://github.com/obra/superpowers) | `scripts/bootstrap_agents.py` |
| 2 | [GitHub Spec Kit](https://github.com/github/spec-kit) | `scripts/bootstrap_agents.py` |
| 3 | [gstack](https://github.com/garrytan/gstack) | `scripts/bootstrap_agents.py` |
| 4 | [Aider](https://github.com/Aider-AI/aider) | `config/harness/agents.json` |
| 5 | [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) | `config/harness/agents.json` |
| 6 | [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | `config/harness/agents.json` |
| 7 | [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent) | `config/harness/agents.json` |
| 8 | [uv](https://docs.astral.sh/uv/) | `uv.lock` |
| 9 | [Ruff](https://docs.astral.sh/ruff/) | `noxfile.py:lint` |
| 10 | [mypy](https://mypy.readthedocs.io/en/stable/) | `noxfile.py:types` |
| 11 | [pre-commit](https://pre-commit.com/) | `.pre-commit-config.yaml` |
| 12 | [pytest](https://docs.pytest.org/en/stable/) | `noxfile.py:tests` |
| 13 | [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) | `noxfile.py:coverage` |
| 14 | [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) | `tests/test_formula_properties.py` |
| 15 | [mutmut](https://mutmut.readthedocs.io/en/latest/) | `pyproject.toml:tool.mutmut` |
| 16 | [pytest-regressions](https://pytest-regressions.readthedocs.io/en/latest/) | `tests/test_output_regression.py` |
| 17 | [Nox](https://nox.thea.codes/en/stable/) | `noxfile.py` |
| 18 | [Pandera](https://pandera.readthedocs.io/en/stable/) | `integrations.data_quality.validate_frame` |
| 19 | [Great Expectations / GX Core](https://docs.greatexpectations.io/docs/core/introduction/) | `integrations.data_quality.validate_with_gx` |
| 20 | [DVC](https://doc.dvc.org/) | `dvc.yaml` |
| 21 | [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/) | `integrations.tracking.record_mlflow_run` |
| 22 | [Evidently](https://docs.evidentlyai.com/introduction) | `integrations.drift.evidently_report` |
| 23 | [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | `integrations.research.time_series_splits` |
| 24 | [sktime](https://www.sktime.net/) | `integrations.research.evaluate_naive_forecast` |
| 25 | [Microsoft Qlib](https://github.com/microsoft/qlib) | `integrations.research` |
| 26 | [Optuna](https://optuna.readthedocs.io/en/stable/) | `integrations.research.tune_train_validation` |
| 27 | [River progressive validation](https://riverml.xyz/latest/api/evaluate/progressive-val-score/) | `integrations.river.make_adaptive_edge_regressor` |
| 28 | [arch 时间序列 bootstrap](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html) | `integrations.research.bootstrap_mean_interval` |
| 29 | [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/) | `benchmarks/test_benchmarks.py` |
| 30 | [ASV](https://asv.readthedocs.io/en/stable/) | `asv.conf.json` |
| 31 | [Memray](https://bloomberg.github.io/memray/) | `noxfile.py:memory` |
| 32 | [Bandit](https://bandit.readthedocs.io/en/latest/) | `config/harness/bandit.yaml` |
| 33 | [pip-audit](https://github.com/pypa/pip-audit) | `noxfile.py:audit` |
| 34 | [Gitleaks](https://github.com/gitleaks/gitleaks) | `noxfile.py:secrets` |
| 35 | [GitHub CodeQL](https://docs.github.com/en/code-security/concepts/code-scanning/codeql/codeql-code-scanning) | [Successful default-setup run on `main` at `609368b`](https://github.com/superwesleyhys-ux/agipot-stock-prediction/actions/runs/35142150961); advanced alternative: `docs/harness/workflows/codeql.yml` |
| 36 | [GitHub Actions](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) | `docs/harness/workflows/tests.yml` |
| 37 | [PyPA build](https://build.pypa.io/en/stable/) | `noxfile.py:distribution` |
| 38 | [Twine](https://twine.readthedocs.io/en/stable/) | `scripts/check_distribution.py` |
| 39 | [Schemathesis](https://schemathesis.readthedocs.io/en/stable/) | `integrations.optional.smoke_openapi` |
| 40 | [Playwright](https://playwright.dev/python/) | `integrations.optional.smoke_browser` |
| 41 | [Testcontainers Python](https://testcontainers-python.readthedocs.io/en/latest/) | `integrations.optional.smoke_container` |
| 42 | [Promptfoo](https://www.promptfoo.dev/docs/intro/) | `config/harness/promptfooconfig.yaml` |

## Optional framework environment

```sh
uv sync --locked --extra harness --extra integrations
uv run playwright install chromium
uv run pytest tests/test_integrations.py -q
npm ci --no-audit --no-fund
PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE_CHECK=1 \
  PROMPTFOO_PYTHON=.venv/bin/python PYTHONPATH=src npm run test:evidence
```

The tests call real Pandera, GX, MLflow, Evidently, sklearn, sktime, Qlib, Optuna,
River, arch, Schemathesis and Chromium APIs with synthetic inputs. Schemathesis
uses a temporary loopback HTTP server; the stock package does not expose a web
API. Testcontainers requires a running Docker daemon and an explicitly supplied,
preloaded image. Promptfoo uses the deterministic shared evidence harness and
does not call an LLM. See the adapter guide for exact examples.
Promptfoo requires Node 22.22.0 or newer. Optional framework tests explicitly skip
missing SDKs or runtime prerequisites; a passing base test run does not establish
that every adapter ran. Read the skip reasons and the verification record.

## Developer agents

```sh
python scripts/bootstrap_agents.py
python scripts/bootstrap_agents.py --executors --build-browser
python scripts/run_agent.py --help
```

Superpowers and gstack are pinned Git checkouts under `.harness/tools/`, linked
into project-local `.agents/skills/`. Spec Kit supplies the `.specify/` scaffold
and project constitution. Skills become discoverable on the next agent turn.
Aider, mini-SWE-agent, OpenHands SDK and RD-Agent use separate Python environments
and explicit task/workspace arguments. The bootstrap installs dependencies but
never invokes a model. Configure each tool's backend yourself; credentials stay
in the local environment. Do not run several editing agents in the same checkout.
`config/harness/agents.json` pins direct versions for bootstrap. Installation
receipts do not establish successful model execution. `doctor` probes the current
Python environment and configured local paths; isolated executor installation
and GitHub-hosted execution have separate receipts. Their own licenses remain
applicable; see [third-party notices](harness/THIRD_PARTY.md).

## Performance and data provenance

```sh
uv sync --locked --extra harness --extra performance --extra data-versioning
uv run nox -s benchmark memory
uv run asv machine --yes
uv run asv run --quick --environment existing:.venv/bin/python --dry-run
uv run dvc repro
```

pytest-benchmark saves local measurements. The ASV command above executes a quick
measurement but `--dry-run` does not save a historical ASV baseline; omit that
flag when intentionally recording one. A single measurement is not a regression
comparison. Memray records the synthetic stock demonstration; its fixed
`reports/memory.bin` output must be moved aside before another `nox -s memory` run.
DVC records source/config/lock inputs and demo outputs in `dvc.lock`, with no
remote data store configured. `.dvcignore` excludes bytecode/cache directories,
and the persistent output setting retains prior isolated demo runs. `dvc repro`
can skip an unchanged stage; `uv run dvc repro --force` requests another run.
New run IDs and event timestamps intentionally differ, so these audit artifacts
are traceable snapshots rather than bit-for-bit deterministic output archives.
Do not put real licensed market data into Git or a public DVC remote.

The recorded `pip-audit` run found **`diskcache==5.6.3`, CVE-2025-69872**, through
the optional DVC environment, with **no fixed version reported**. This finding
is retained, not added to an ignore list or reported as a clean security scan.
After installing `data-versioning`, audit that exact environment with
`uv run --no-sync pip-audit`; a failing exit remains a finding. A clean audit of a
smaller environment without DVC does not resolve it. See the
[verification record](harness/VERIFICATION.md) for the recorded scope and result.

## Quality and security scope

`nox` is configured for lint, Python 3.11/3.12/3.13 tests, formula properties,
scoped typing, and sdist/wheel installation checks. Missing interpreters or
prerequisites must not be counted as executed checks. `uv sync --locked` uses
the repository lock; Nox's isolated `session.install` calls currently resolve
the declared package ranges independently and do not consume `uv.lock`.
Ruff checks parse/name/unused-local errors;
mypy currently checks `research/contracts.py` and `research/cost_model.py` only.
Mutation testing targets `research/cost_model.py`; inspect surviving mutations,
not only the process exit code. Explicit mutant names rerun cached verdicts:

```sh
uv run mutmut run 'agipot_stock_prediction.research.cost_model.*'
uv run mutmut results
uv run pre-commit install
uv run pre-commit run --all-files
uv run nox -s audit
```

Bandit excludes the subprocess-import/fixed-command rules B404/B603/B607;
three B105 false positives on boolean `oos_pass` values are documented inline.
Gitleaks is a separate binary: add it to PATH and run `nox -s secrets` for history;
scan the staged working tree separately before publication. Neither tool proves
absence of vulnerabilities.

## GitHub automation

CodeQL default setup is enabled through GitHub's supported repository API.
[Run 35142150961](https://github.com/superwesleyhys-ux/agipot-stock-prediction/actions/runs/35142150961)
completed successfully on `main` at commit
`609368b33a723e99bd679cab45b35b9f5dd62cbd`; it is evidence for that revision, not
for later unmerged harness changes. The reviewed Python test matrix is provided at
[`harness/workflows/tests.yml`](harness/workflows/tests.yml). This installation's
GitHub OAuth grant lacks `workflow`; publishing a custom workflow requires that
scope. Once granted, run `python scripts/enable_ci.py`, review, commit and push.
The CodeQL advanced template is an alternative only: disable default setup before
using it. The templates are not reported as executed test CI.
