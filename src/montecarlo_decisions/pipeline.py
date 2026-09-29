"""End-to-end pipeline: history -> models -> counterfactuals -> Monte Carlo -> artifacts.

It is split in two stages so the web app can prepare once and simulate on demand:

* :func:`prepare_case` builds the data, fits every model and precomputes the
  incremental-value tensor (the expensive part, ~tens of seconds);
* :func:`run_simulation` draws the futures, summarises them and writes every artifact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from montecarlo_decisions.config import SEED, ArtifactPaths
from montecarlo_decisions.counterfactuals import (
    LeverSample,
    add_bootstrap_intervals,
    lever_effects,
    lever_samples,
    value_lifts,
)
from montecarlo_decisions.models import (
    ValueModel,
    bootstrap_rows,
    evaluate_models,
    fit_value_model,
    with_time_index,
)
from montecarlo_decisions.report import write_report
from montecarlo_decisions.scenarios import (
    DECISION_KEYS,
    ScenarioConfig,
    estimate_ads_response,
    load_config,
    scenario_expected_values,
)
from montecarlo_decisions.simulation import DeltaTensor, ProgressCallback, simulate, summarize
from montecarlo_decisions.synthetic import N_ROWS, generate_dataset
from montecarlo_decisions.validation import Check, run_checks

DEFAULT_SIMULATIONS = 10_000
logger = logging.getLogger(__name__)


@dataclass
class PreparedCase:
    """Everything the simulation needs, computed once."""

    config: ScenarioConfig
    history: pd.DataFrame
    model: ValueModel
    model_report: dict
    uplift: pd.DataFrame
    ads_response: pd.DataFrame
    base: pd.DataFrame
    deltas: DeltaTensor
    seed: int = SEED


@dataclass
class PipelineResult:
    prepared: PreparedCase
    simulations: pd.DataFrame
    summary: pd.DataFrame
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def planning_base(history: pd.DataFrame, config: ScenarioConfig, seed: int) -> pd.DataFrame:
    """Bootstrap today's opportunity mix up to the volume expected over the horizon."""
    window = history[history["date"] >= config.simulation.base_window_start]
    if window.empty:
        raise ValueError("No opportunities in the base window; check base_window_start")
    months = window["month"].nunique()
    horizon_volume = round(len(window) / months * config.simulation.horizon_months)
    return window.sample(horizon_volume, replace=True, random_state=seed).reset_index(drop=True)


def replicate_deltas(
    model: ValueModel,
    base: pd.DataFrame,
    config: ScenarioConfig,
    ads_response: pd.DataFrame,
) -> np.ndarray:
    """Incremental expected value, shape (n_decisions, n_opportunities), for one model."""
    baseline, scenarios = scenario_expected_values(model, base, config, ads_response)
    return np.stack([scenarios[key] - baseline for key in DECISION_KEYS])


def _fit_replicate(
    history: pd.DataFrame,
    rows: np.ndarray,
    seed: int,
    base: pd.DataFrame,
    config: ScenarioConfig,
    ads_response: pd.DataFrame,
    samples: list[LeverSample],
) -> tuple[np.ndarray, np.ndarray]:
    """Refit on one bootstrap resample; return its decision deltas and lever lifts."""
    model = fit_value_model(history.iloc[rows], seed=seed)
    return replicate_deltas(model, base, config, ads_response), value_lifts(model, samples)


def prepare_case(
    config: ScenarioConfig | None = None,
    seed: int = SEED,
    n_rows: int = N_ROWS,
    n_bootstrap_models: int | None = None,
    n_jobs: int = -1,
) -> PreparedCase:
    """Build the history, fit and validate the models, precompute the delta tensor.

    Bootstrap replicates are independent and run in parallel (``n_jobs`` as in joblib).
    Their rows and seeds are fixed up front, so results do not depend on ``n_jobs``.
    """
    config = config or load_config()
    replicates = n_bootstrap_models or config.simulation.n_bootstrap_models
    extra_cost = config.levers["new_product"]["extra_cost_per_offer_eur"]

    logger.info("Generating %s synthetic opportunities", f"{n_rows:,}")
    history = with_time_index(generate_dataset(n_rows, seed))
    logger.info("Evaluating models on a temporal holdout")
    model_report = evaluate_models(history, seed)
    logger.info("Fitting value models (time-controlled and naive)")
    model = fit_value_model(history, seed)
    naive = fit_value_model(history, seed, time_control=False)
    uplift = lever_effects(history, model, naive, extra_cost)

    ads_response = estimate_ads_response(history)
    base = planning_base(history, config, seed)
    samples = lever_samples(history, extra_cost, seed=seed)
    logger.info("Fitting %d bootstrap replicates", replicates)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_replicate)(history, rows, seed + r, base, config, ads_response, samples)
        for r, rows in enumerate(bootstrap_rows(len(history), replicates, seed))
    )
    deltas = np.stack([delta for delta, _ in results])
    if not np.isfinite(deltas).all():
        raise ValueError("Non-finite incremental values; check the scenario inputs")
    uplift = add_bootstrap_intervals(uplift, np.stack([lifts for _, lifts in results]))
    return PreparedCase(
        config,
        history,
        model,
        model_report,
        uplift,
        ads_response,
        base,
        DeltaTensor(deltas, DECISION_KEYS),
        seed,
    )


def run_simulation(
    prepared: PreparedCase,
    paths: ArtifactPaths,
    n_simulations: int = DEFAULT_SIMULATIONS,
    progress: ProgressCallback | None = None,
    pace_seconds: float = 0.0,
) -> PipelineResult:
    labels = prepared.config.labels()
    logger.info("Simulating %s futures", f"{n_simulations:,}")
    simulations = simulate(
        prepared.deltas,
        prepared.config,
        n_simulations,
        seed=prepared.seed,
        progress=progress,
        pace_seconds=pace_seconds,
    )
    summary = summarize(simulations, labels)
    checks = run_checks(
        prepared.history, prepared.model_report, prepared.uplift, simulations, summary
    )
    result = PipelineResult(prepared, simulations, summary, checks)
    write_artifacts(result, paths)
    return result


def write_artifacts(result: PipelineResult, paths: ArtifactPaths) -> None:
    prepared = result.prepared
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    prepared.history.drop(columns=["month_index"]).to_csv(paths.dataset, index=False)
    paths.model_report.write_text(json.dumps(prepared.model_report, indent=2), encoding="utf-8")
    prepared.uplift.to_csv(paths.uplift, index=False)
    prepared.ads_response.reset_index().to_csv(paths.ads_regimes, index=False)
    result.simulations.to_csv(paths.simulations, index=False, float_format="%.2f")
    result.summary.to_csv(paths.summary, index=False)
    paths.checks.write_text(
        json.dumps([check.as_dict() for check in result.checks], indent=2), encoding="utf-8"
    )
    write_report(result, paths.report)


def run_pipeline(
    paths: ArtifactPaths,
    n_simulations: int = DEFAULT_SIMULATIONS,
    seed: int = SEED,
    config: ScenarioConfig | None = None,
    progress: ProgressCallback | None = None,
    pace_seconds: float = 0.0,
    n_rows: int = N_ROWS,
    n_bootstrap_models: int | None = None,
    n_jobs: int = -1,
) -> PipelineResult:
    prepared = prepare_case(config, seed, n_rows, n_bootstrap_models, n_jobs)
    return run_simulation(prepared, paths, n_simulations, progress, pace_seconds)
