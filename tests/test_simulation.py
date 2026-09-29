import dataclasses

import numpy as np
import pandas as pd
import pytest

from montecarlo_decisions.scenarios import DECISION_KEYS
from montecarlo_decisions.simulation import DeltaTensor, lognormal_params, simulate, summarize


def _tensor(values_per_decision: list[float], n_opportunities: int = 50) -> DeltaTensor:
    values = np.array(values_per_decision)[None, :, None] * np.ones((2, 1, n_opportunities))
    return DeltaTensor(values, DECISION_KEYS)


def _without_risk(config):
    decisions = {
        key: dataclasses.replace(
            decision,
            failure_probability=0.0,
            realization_mean=1.0,
            realization_sd=0.0,
            cost_overrun_sd=0.0,
        )
        for key, decision in config.decisions.items()
    }
    simulation = dataclasses.replace(config.simulation, volume_cv=0.0)
    return dataclasses.replace(config, decisions=decisions, simulation=simulation)


def test_lognormal_params_match_moments():
    mu, sigma = lognormal_params(0.9, 0.2)
    draws = np.random.default_rng(0).lognormal(mu, sigma, 400_000)
    assert draws.mean() == pytest.approx(0.9, rel=0.01)
    assert draws.std() == pytest.approx(0.2, rel=0.02)
    assert lognormal_params(1.0, 0.0) == (0.0, 0.0)


def test_delta_tensor_validates_shape():
    with pytest.raises(ValueError, match="shape"):
        DeltaTensor(np.zeros((2, 3, 5)), DECISION_KEYS)


def test_simulation_is_deterministic(prepared):
    first = simulate(prepared.deltas, prepared.config, 300, seed=11)
    second = simulate(prepared.deltas, prepared.config, 300, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 300 * len(DECISION_KEYS)


def test_common_random_numbers_across_decisions(prepared):
    sims = simulate(prepared.deltas, prepared.config, 200, seed=3)
    shared = sims.groupby("simulation")[["model_replicate", "volume_factor"]].nunique()
    assert (shared == 1).all().all()


def test_zero_effect_only_costs_money(config):
    sims = simulate(_tensor([0.0, 0.0, 0.0, 0.0]), _without_risk(config), 100)
    fixed = sims["decision"].map({k: d.fixed_cost_eur for k, d in config.decisions.items()})
    np.testing.assert_allclose(sims["incremental_profit_eur"], -fixed)


def test_certain_world_reproduces_the_expected_delta(config):
    per_opportunity, n = 10.0, 50
    certain = _without_risk(config)
    sims = simulate(_tensor([per_opportunity] * 4, n), certain, 200)
    summary = summarize(sims).set_index("decision")
    expected = per_opportunity * n - config.decisions["funnel"].fixed_cost_eur
    assert summary.loc["funnel", "expected_profit_eur"] == pytest.approx(expected)
    assert summary.loc["funnel", "std_eur"] == pytest.approx(0.0, abs=1e-6)


def test_failed_initiatives_lose_their_fixed_cost(config):
    decisions = {
        key: dataclasses.replace(d, failure_probability=1.0, cost_overrun_sd=0.0)
        for key, d in config.decisions.items()
    }
    sims = simulate(_tensor([100.0] * 4), dataclasses.replace(config, decisions=decisions), 50)
    assert not sims["succeeded"].any()
    assert (sims["incremental_profit_eur"] == -sims["cost_eur"]).all()


def test_progress_is_reported_until_completion(prepared):
    seen: list[tuple[int, int]] = []
    simulate(
        prepared.deltas,
        prepared.config,
        1_200,
        progress=lambda done, total, frame: seen.append((done, total)),
    )
    counts = [done for done, _ in seen]
    assert counts == sorted(counts)
    assert seen[-1] == (1_200, 1_200)


def test_summary_metrics_are_consistent(result):
    summary = result.summary
    assert summary["expected_profit_eur"].is_monotonic_decreasing
    assert summary["ranking"].tolist() == [1, 2, 3, 4]
    ordered = summary[["p5_eur", "p10_eur", "p50_eur", "p90_eur", "p95_eur"]]
    assert (ordered.diff(axis=1).iloc[:, 1:] >= 0).all().all()
    assert (summary["cvar_5_eur"] <= summary["p5_eur"]).all()
    assert summary["probability_best"].sum() == pytest.approx(1.0)
    assert summary["probability_loss"].between(0, 1).all()


def test_breakeven_failure_probability_zeroes_expected_profit(result):
    sims = result.simulations
    for row in result.summary.itertuples():
        if not 0 < row.breakeven_failure_probability < 1:
            continue
        group = sims[sims["decision"] == row.decision]
        at_breakeven = (1 - row.breakeven_failure_probability) * group[
            "gross_gain_eur"
        ].mean() - group["cost_eur"].mean()
        assert at_breakeven == pytest.approx(0.0, abs=1e-6)


def test_summarize_empty_frame():
    assert summarize(pd.DataFrame()).empty


@pytest.mark.slow
def test_preparation_does_not_depend_on_parallelism(config):
    from montecarlo_decisions.pipeline import prepare_case

    serial = prepare_case(config, n_rows=4_000, n_bootstrap_models=2, n_jobs=1)
    parallel = prepare_case(config, n_rows=4_000, n_bootstrap_models=2, n_jobs=2)
    np.testing.assert_array_equal(serial.deltas.values, parallel.deltas.values)
    pd.testing.assert_frame_equal(serial.uplift, parallel.uplift)
