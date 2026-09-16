# Integration verification — 2026-09-16

This is implementation evidence on synthetic/local inputs. It is not evidence
of profitable stock prediction, production readiness, complete semantic truth,
or a full-project security/type audit. Installation, configuration and actual
execution are deliberately distinguished.

## Executed checks

| Check | Result and scope |
|---|---|
| Locked full Python environment | **320 passed, 0 skipped**. `uv sync --locked --extra harness --extra integrations`, Python 3.12, actual Chromium enabled. |
| Nox Python matrix | **307 passed, 13 optional skips per interpreter** on Python 3.11.15, 3.12.6 and 3.13.13. Nox installs declared ranges independently of uv.lock. |
| A–F demo | All six PASS; separate JSON artifacts and SHA-256 receipts; expected failed/aborted trials remain in E's verified log. |
| Promptfoo / npm | Fixed Promptfoo **0.122.1**; ordinary clean `npm ci`, **4 cases passed**, full `npm audit`: **0 vulnerabilities**. |
| Optional adapters | **32 passed** with real SDKs. Browser uses a local synthetic report; Schemathesis uses a temporary loopback service. Docker execution is not included. |
| Formula mutation testing | **119 mutants: 99 killed, 20 survived, 0 timed out**. All surviving changes affect error-message text only; arithmetic and input rejection mutations were caught. See [review](mutation-review.json). |
| Hypothesis + regressions | Generated numerical/unit/ensemble properties, batch/stream parity and saved report contract pass. |
| Coverage | **75%** combined line/branch coverage in the development environment, 313 passed/7 optional skips. Coverage is reported, not advertised as complete. |
| Ruff / mypy | Pass. Ruff scope is configured parse/name/unused-local rules; mypy checks two declared research modules only. |
| Build / Twine / installed wheel | sdist and wheel build, metadata/archive checks pass; import and both CLIs run from outside the checkout. Wheel-only tests skip unavailable optional SDKs. |
| pytest-benchmark | One local baseline passes; online prediction mean about 6.60 µs on this run. No before/after regression conclusion. |
| ASV | Two quick-mode measurements run against the installed package: prediction and peak memory. Dry-run does not save historical comparison results. |
| Memray | Synthetic stock demo profiled successfully; peak tracked allocation about 41.65 MB on this machine. |
| DVC | Actual `dvc repro` generated the checked-in `dvc.lock`; bytecode is excluded and run folders are persisted. No remote configured. |
| Bandit | No remaining findings in configured source scope; three boolean `oos_pass` false positives are annotated. B404/B603/B607 are excluded for the explicit subprocess runners. |
| pip-audit: core/harness environment | No known dependency vulnerabilities found. The unpublished project itself is not available in the advisory database. |
| Gitleaks | Redacted source/publication-snapshot and Git-history scans run before publishing. |
| pre-commit | Installed locally; Ruff and formula contract hooks pass. |
| GitHub CodeQL | Default setup enabled; [initial run passed](https://github.com/superwesleyhys-ux/agipot-stock-prediction/actions/runs/35142150961) on the preceding main revision. New pushes are scanned independently. |

## Developer agents and prerequisites

Superpowers project skills and Spec Kit scaffold are installed. gstack's pinned
Bun build was executed locally; its project browser binary was produced. Four
executor environments are separate from the stock runtime. CLI help/SDK wiring
checks do not call a model and do not prove that a configured model completes
an editing task. User model credentials/backends are not bundled. Aider and mini-SWE-agent help
commands passed; OpenHands SDK/tools 1.48.0 imports and signatures matched the
launcher. RD-Agent 0.8.0 main and `fin_factor` help passed after pinning compatible
`pydantic-ai-slim` 1.107.5 and `mcp` 1.30.0; its `uv pip check` also passed.

- Testcontainers has a real adapter, but the local Docker daemon is unavailable;
  a caller-selected preloaded image and running daemon are still required.
- Custom GitHub Python test CI is a reviewed template, awaiting a `workflow`
  OAuth grant. CodeQL default setup is already enabled independently.
- Point-in-time and evidence contracts check supplied availability and annotations;
  they cannot establish correctness of metadata the caller fabricated.

## Dependency finding retained

The optional DVC environment includes **diskcache 5.6.3**, with
**CVE-2025-69872 / GHSA-w8v5-vhqr-4h9v / PYSEC-2026-2447**. The audit returned
no fixed version. It concerns default pickle deserialization when an attacker
can write to a cache directory. This finding is not ignored or called a passing
audit. Keep DVC caches private to trusted local processes; do not ingest cache
files from untrusted users. The core stock runtime does not depend on diskcache.

See [integration APIs and tested versions](../INTEGRATIONS.md) and
[reproduction commands](../HARNESSES.md). Raw logs and runtime artifacts remain
under local `reports/` and temporary verification environments; secrets, market
data, model credentials and machine-specific runtime directories are excluded
from the release.
