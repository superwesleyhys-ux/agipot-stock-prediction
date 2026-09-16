# AGIPOT development harness

Use one coordinating coding agent; isolate alternative executors in their own workspaces.
Task scope comes from the user. The installed skill packs provide reusable procedures;
they do not grant authority to publish data, credentials or trading instructions.

Before completing code changes, run the relevant focused tests, then the shared checks:

```sh
uv sync --extra dev --extra harness --extra data --locked
uv run agipot-harness run lint
uv run agipot-harness run tests
uv run agipot-harness demo
```

Preserve existing scoring cutoff differences. Training, preprocessing, data revisions,
model availability and label maturity are separate time constraints. Do not claim
OOS/profitability from implementation tests or rename heuristic PBO/DSR as formal estimates.
Record failed and aborted trials. Generated summaries are display content, not evidence.

Agent skill setup: `python scripts/bootstrap_agents.py`. It installs project-local
Superpowers, Spec Kit and gstack skills; they are available on the next turn. Alternative
coding/research executors use `scripts/run_agent.py` with an explicit workspace.

Quality commands, framework adapters and actual verification states are documented in
`docs/HARNESSES.md`. Missing Docker, browser binaries, service URLs, model credentials
or workflow permission must remain explicit blockers, not silent successful checks.
