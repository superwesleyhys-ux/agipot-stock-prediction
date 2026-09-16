# Formula and validation extraction

`SOURCES.json` records the original AGIPOT source path and SHA-256 for every
copied or extracted file. Paths are relative to the original repository and the
new package; they do not expose workstation paths. New code is identified below.

## Scope

All original pure formula modules are included: quality, moat, owner earnings,
intrinsic value, value traps, trend confirmation, universe, liquidity, ensemble,
cost, sizing, feedback, hedging, general/workout/control sleeves, price collars,
position adjustment, kill-switch scoring, and promotion/trade-policy scoring.
`modes.py` contains enum labels and pure enum helpers only. Research contracts
retain only `ResearchDataError` and `finite_number`. No data-store, broker,
account-access, credential, authentication, network, or submission implementation
is included. Terms such as account equity and activation-token validity remain
caller-supplied numeric or boolean evidence for offline formulas.

## Changes from source

- Imports point to this package. New `_validation.py` checks declared input
  types, finite numbers, numeric sequences/mappings, and explicit booleans before
  running formulas; missing/NaN evidence cannot silently pass a comparison.
- Ensemble uses stable softmax with iterative redistribution to retain both
  unit total weight and the actual per-skill cap. Infeasible caps raise an error.
  Missing reliability values block the gate.
- Data-integrity scoring requires nonempty, equally sized price/quantity data
  and nonnegative quote ages. Quality/moat/liquidity/hedging gates explicitly
  block missing histories or observations instead of passing with defaults.
  A moat price proxy remains visible as a score but cannot pass the fundamental
  evidence gate.
- Domain checks reject invalid sizing variance, weight limits, pricing limits,
  history-window lengths, valuation horizons, negative costs, and invalid
  position inputs. Sell-policy scoring requires a positive owned quantity.
- `walk_forward_factory` now uses all declared calendar periods and creates
  separate train, validation, and test partitions. It omits incomplete terminal
  periods conservatively. It does not fit models, implement purging/embargo,
  verify a data vendor, or establish that labels are out of sample.
- `overfit_guard` retains the original heuristic computations but reports
  `heuristic_only`, explicitly marks formal PBO/DSR as uncomputed, and rejects
  incomplete/nonfinite trial evidence. Its historical `walk_forward_pass_rate`
  field is only the supplied trial mandate-pass fraction. No formal inference
  should be made from this result.
- `result_ranker` preserves a true zero-percent drawdown, validates required
  metrics, and rejects nonfinite evidence.

`tests/test_formulas.py` and `tests/test_validation.py` use only fabricated
numbers and sessions to verify these behaviors. Their outputs are software
checks, not performance estimates or trading validation.

These modules use only the Python standard library at runtime.
