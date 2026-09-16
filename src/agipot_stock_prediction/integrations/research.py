"""Optional research frameworks operating on supplied, local observations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ._optional import disable_telemetry, finite_values, require


def time_series_splits(n_samples: int, *, n_splits: int = 5, gap: int = 0,
                       test_size: int | None = None) -> tuple[dict[str, list[int]], ...]:
    """Use sklearn TimeSeriesSplit. ``gap`` is a row count, not label-time purge."""
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be positive")
    if type(gap) is not int or gap < 0:
        raise ValueError("gap must be a nonnegative integer")
    model_selection = require("sklearn.model_selection", "scikit-learn")
    splitter = model_selection.TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    return tuple({"train": train.tolist(), "validation": test.tolist()}
                 for train, test in splitter.split(range(n_samples)))


def evaluate_naive_forecast(values: Sequence[float], *, initial_window: int,
                            horizon: int = 1, step_length: int = 1) -> Any:
    """Run sktime expanding-window evaluation of a last-observation baseline.

    This baseline refits on each fold and returns sktime's MAE/timing DataFrame.
    It is not an evaluation of the AGIPOT scoring or edge models.
    """
    series_values = finite_values(values, name="forecast series", minimum=3)
    if any(type(value) is not int or value < 1 for value in (initial_window, horizon, step_length)):
        raise ValueError("initial_window, horizon, and step_length must be positive integers")
    if initial_window + horizon > len(series_values):
        raise ValueError("not enough observations for a full forecast fold")
    pd = require("pandas")
    naive = require("sktime.forecasting.naive", "sktime")
    split = require("sktime.split", "sktime")
    evaluation = require("sktime.forecasting.model_evaluation", "sktime")
    metrics = require("sktime.performance_metrics.forecasting", "sktime")
    cv = split.ExpandingWindowSplitter(initial_window=initial_window,
                                       fh=list(range(1, horizon + 1)), step_length=step_length)
    return evaluation.evaluate(naive.NaiveForecaster(strategy="last"), cv=cv,
                               y=pd.Series(series_values), strategy="refit",
                               scoring=metrics.MeanAbsoluteError(), error_score="raise")


def tune_train_validation(
    rows: Sequence[Any], *, train_end: int, validation_end: int,
    objective: Callable[[Any, Sequence[Any], Sequence[Any]], float],
    n_trials: int = 20, seed: int = 0, direction: str = "minimize",
) -> dict[str, Any]:
    """Run Optuna using only [0:train_end] and [train_end:validation_end].

    The final suffix is held out and never passed to the objective. Each trial
    gets fresh copies so its mutations cannot affect later trials. The callback
    must itself be pure: Python closures can access external data, which this
    adapter cannot sandbox. No final holdout score is calculated here.
    """
    if not (type(train_end) is int and type(validation_end) is int and 0 < train_end < validation_end < len(rows)):
        raise ValueError("require nonempty chronological train, validation, and final holdout suffix")
    if type(n_trials) is not int or n_trials < 1 or direction not in {"minimize", "maximize"}:
        raise ValueError("n_trials must be positive and direction minimize/maximize")
    optuna = require("optuna")
    train = deepcopy(rows[:train_end])
    validation = deepcopy(rows[train_end:validation_end])
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))

    def measured(trial: Any) -> float:
        return finite_values([objective(trial, deepcopy(train), deepcopy(validation))], name="objective")[0]

    study.optimize(measured, n_trials=n_trials, n_jobs=1, catch=(ValueError, ArithmeticError))
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    return {"framework": "optuna", "best_params": dict(study.best_params) if completed else None,
            "best_value": study.best_value if completed else None,
            "status": "completed" if completed else "all_trials_failed",
            "trials": [{"number": trial.number, "state": trial.state.name, "value": trial.value,
                        "params": dict(trial.params)} for trial in study.trials],
            "train_end": train_end, "validation_end": validation_end,
            "final_holdout_evaluated": False}


def bootstrap_mean_interval(returns: Sequence[float], *, block_size: int,
                            reps: int = 1000, confidence: float = 0.95, seed: int = 0) -> dict[str, Any]:
    """Stationary-bootstrap interval for mean decimal returns (0.01 = 1%)."""
    numbers = finite_values(returns, name="returns", minimum=3)
    if type(block_size) is not int or not 1 <= block_size <= len(numbers):
        raise ValueError("block_size must be an integer between 1 and sample count")
    if type(reps) is not int or reps < 10 or not 0 < confidence < 1:
        raise ValueError("reps must be >= 10 and confidence in (0, 1)")
    np = require("numpy")
    bootstrap = require("arch.bootstrap", "arch")
    sampler = bootstrap.StationaryBootstrap(block_size, np.asarray(numbers), seed=seed)
    interval = sampler.conf_int(np.mean, reps=reps, method="percentile", size=confidence)
    return {"framework": "arch", "method": "stationary_bootstrap_percentile_mean",
            "mean": float(np.mean(numbers)), "lower": float(interval[0, 0]), "upper": float(interval[1, 0]),
            "confidence": confidence, "block_size": block_size, "reps": reps, "seed": seed,
            "sample_count": len(numbers), "return_unit": "decimal", "formal_pbo_computed": False,
            "formal_dsr_computed": False}


def import_qlib_predictions(source: str | Path | Any) -> Any:
    """Normalize Qlib prediction exports to [datetime, instrument] + score.

    Accepts an in-memory Series/DataFrame or local CSV. No pickle loading, market
    data download, Qlib init, or implicit provider access is performed.
    """
    pd = require("pandas")
    if isinstance(source, (str, Path)):
        if "://" in str(source):
            raise ValueError("Qlib prediction source must be an existing local CSV")
        frame = pd.read_csv(Path(source))
    elif isinstance(source, pd.Series):
        frame = source.rename("score").to_frame()
    else:
        frame = source.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    required = {"datetime", "instrument", "score"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("Qlib export requires datetime, instrument, and score with nonempty data")
    frame = frame.loc[:, ["datetime", "instrument", "score"]].copy()
    frame["datetime"] = pd.to_datetime(frame.datetime, utc=True, errors="raise")
    frame["score"] = finite_values(frame.score, name="prediction score")
    if frame.instrument.isna().any() or not frame.instrument.map(lambda item: isinstance(item, str) and bool(item.strip())).all():
        raise ValueError("instrument values must be nonempty strings")
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError("duplicate Qlib prediction identity")
    return frame.sort_values(["datetime", "instrument"]).set_index(["datetime", "instrument"])


def qlib_risk_analysis(returns: Sequence[float], *, annualization: int = 252) -> Mapping[str, Any]:
    """Call Qlib risk_analysis on supplied decimal simple returns.

    Uses Qlib's arithmetic/sum accumulation convention, not compounded CAGR.
    ``annualization`` is an explicit periods-per-year factor. Undefined ratio
    metrics are represented as null, not silently labelled as passing.
    """
    numbers = finite_values(returns, name="decimal returns", minimum=2)
    if type(annualization) is not int or annualization < 1:
        raise ValueError("annualization must be a positive integer")
    pd = require("pandas")
    disable_telemetry()
    evaluator = require("qlib.contrib.evaluate", "pyqlib")
    result = evaluator.risk_analysis(pd.Series(numbers), N=annualization, freq=None)
    import math
    metrics = {str(key): float(value) if math.isfinite(float(value)) else None
               for key, value in result["risk"].items()}
    return {"framework": "qlib", "metrics": metrics, "return_unit": "decimal",
            "accumulation": "sum", "annualization": annualization,
            "sample_count": len(numbers), "out_of_sample_verified": False}
