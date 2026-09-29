"""Vectorised Monte Carlo over the uncertainty that matters for the decision.

Each simulated future draws, in this order:

1. **Model uncertainty**: one of ``R`` bootstrap refits of the value models.
2. **Opportunity mix**: a bootstrap resample of today's opportunities (multinomial
   counts over the base sample), shared by every decision in that future.
3. **Demand volume**: a multiplicative shock on the number of opportunities.
4. **Execution risk** (per decision, independent): whether the initiative ships at all,
   how much of the estimated effect materialises, and the cost overrun.

Steps 1-3 are common random numbers: all decisions are evaluated on the same draw, so
differences between them are not sampling noise. Irreducible per-opportunity noise
(did *this* lead buy?) is not simulated; over thousands of opportunities it adds little
to the variance of the *incremental* profit, which is what the decision depends on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from montecarlo_decisions.config import SEED
from montecarlo_decisions.scenarios import DECISION_KEYS, Decision, ScenarioConfig

ProgressCallback = Callable[[int, int, pd.DataFrame], None]
SIMULATION_COLUMNS = [
    "simulation",
    "decision",
    "model_replicate",
    "volume_factor",
    "succeeded",
    "gross_gain_eur",
    "cost_eur",
    "incremental_profit_eur",
]


@dataclass(frozen=True)
class DeltaTensor:
    """Incremental expected value per (model replicate, decision, base opportunity)."""

    values: np.ndarray  # shape (n_replicates, n_decisions, n_opportunities)
    decisions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[1] != len(self.decisions):
            raise ValueError("values must have shape (replicates, decisions, opportunities)")


def lognormal_params(mean: float, sd: float) -> tuple[float, float]:
    """(mu, sigma) of a lognormal with the given arithmetic mean and standard deviation."""
    if sd == 0:
        return float(np.log(mean)), 0.0
    sigma2 = np.log1p((sd / mean) ** 2)
    return float(np.log(mean) - sigma2 / 2), float(np.sqrt(sigma2))


def _execution_draws(
    rng: np.random.Generator, decision: Decision, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    succeeded = rng.random(size) >= decision.failure_probability
    mu, sigma = lognormal_params(decision.realization_mean, decision.realization_sd)
    realization = rng.lognormal(mu, sigma, size)
    mu_cost, sigma_cost = lognormal_params(1.0, decision.cost_overrun_sd)
    cost = decision.fixed_cost_eur * rng.lognormal(mu_cost, sigma_cost, size)
    return succeeded, realization, cost


def simulate(
    deltas: DeltaTensor,
    config: ScenarioConfig,
    n_simulations: int,
    seed: int = SEED,
    progress: ProgressCallback | None = None,
    pace_seconds: float = 0.0,
) -> pd.DataFrame:
    """Run ``n_simulations`` futures; one output row per (simulation, decision).

    ``pace_seconds`` stretches the run to roughly that wall-clock duration so a live
    dashboard can be followed during a demo. It never changes the results.
    """
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")
    rng = np.random.default_rng(seed)
    n_replicates, n_decisions, n_opportunities = deltas.values.shape
    uniform = np.full(n_opportunities, 1 / n_opportunities)
    volume_cv = config.simulation.volume_cv
    chunk_size = config.simulation.chunk_size
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []

    done = 0
    while done < n_simulations:
        size = min(chunk_size, n_simulations - done)
        replicate = rng.integers(0, n_replicates, size)
        counts = rng.multinomial(n_opportunities, uniform, size=size).astype(float)
        volume = np.clip(rng.normal(1.0, volume_cv, size), 0.2, None)

        raw = np.empty((size, n_decisions))
        for r in np.unique(replicate):
            rows = replicate == r
            raw[rows] = counts[rows] @ deltas.values[r].T
        raw *= volume[:, None]

        simulation_ids = np.arange(done + 1, done + size + 1)
        for j, key in enumerate(deltas.decisions):
            succeeded, realization, cost = _execution_draws(rng, config.decisions[key], size)
            gross = raw[:, j] * realization
            frames.append(
                pd.DataFrame(
                    {
                        "simulation": simulation_ids,
                        "decision": key,
                        "model_replicate": replicate,
                        "volume_factor": volume,
                        "succeeded": succeeded,
                        "gross_gain_eur": gross,
                        "cost_eur": cost,
                        "incremental_profit_eur": np.where(succeeded, gross, 0.0) - cost,
                    }
                )
            )
        done += size

        if progress is not None:
            progress(done, n_simulations, pd.concat(frames, ignore_index=True))
        if pace_seconds > 0:
            wait = pace_seconds * done / n_simulations - (time.perf_counter() - started)
            if wait > 0:
                time.sleep(wait)

    result = pd.concat(frames, ignore_index=True)[SIMULATION_COLUMNS]
    return result.sort_values(["simulation", "decision"], kind="stable").reset_index(drop=True)


def summarize(simulations: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Risk/return metrics per decision, ranked by expected incremental profit.

    * ``expected_roi`` = E[profit] / E[cost] (a ratio of means, not a mean of ratios).
    * ``cvar_5_eur`` = mean profit over the worst 5% of futures (expected shortfall).
    * ``probability_best`` = share of futures where the decision beats all others.
    * ``breakeven_failure_probability`` = failure probability at which the expected
      profit would be zero, holding everything else fixed.
    * ``mc_standard_error_eur`` = Monte Carlo standard error of the expected profit.
    """
    if simulations.empty:
        return pd.DataFrame()
    wide = simulations.pivot(
        index="simulation", columns="decision", values="incremental_profit_eur"
    )
    best = wide.idxmax(axis=1).value_counts(normalize=True)

    rows = []
    for key, group in simulations.groupby("decision", sort=False):
        profit = group["incremental_profit_eur"].to_numpy()
        worst = np.sort(profit)[: max(1, int(np.ceil(0.05 * len(profit))))]
        gain_if_shipped = group["gross_gain_eur"].mean()
        mean_cost = group["cost_eur"].mean()
        breakeven = 1 - mean_cost / gain_if_shipped if gain_if_shipped > 0 else 0.0
        rows.append(
            {
                "decision": key,
                "label": (labels or {}).get(key, key),
                "expected_profit_eur": profit.mean(),
                "std_eur": profit.std(ddof=1) if len(profit) > 1 else 0.0,
                "p5_eur": np.percentile(profit, 5),
                "p10_eur": np.percentile(profit, 10),
                "p50_eur": np.percentile(profit, 50),
                "p90_eur": np.percentile(profit, 90),
                "p95_eur": np.percentile(profit, 95),
                "cvar_5_eur": worst.mean(),
                "probability_loss": float((profit < 0).mean()),
                "probability_best": float(best.get(key, 0.0)),
                "expected_cost_eur": mean_cost,
                "expected_roi": profit.mean() / mean_cost,
                "breakeven_failure_probability": float(np.clip(breakeven, 0.0, 1.0)),
                "mc_standard_error_eur": (
                    profit.std(ddof=1) / np.sqrt(len(profit)) if len(profit) > 1 else 0.0
                ),
                "n_simulations": len(profit),
            }
        )
    summary = pd.DataFrame(rows).sort_values("expected_profit_eur", ascending=False)
    summary.insert(0, "ranking", np.arange(1, len(summary) + 1))
    return summary.reset_index(drop=True)


__all__ = ["DECISION_KEYS", "DeltaTensor", "lognormal_params", "simulate", "summarize"]
