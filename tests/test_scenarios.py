import copy
import dataclasses
import tomllib

import numpy as np
import pytest

from montecarlo_decisions import scenarios
from montecarlo_decisions.config import SCENARIOS_PATH


@pytest.fixture
def raw_config() -> dict:
    with SCENARIOS_PATH.open("rb") as handle:
        return tomllib.load(handle)


def test_default_config_is_valid():
    config = scenarios.load_config()
    assert tuple(config.decisions) == scenarios.DECISION_KEYS
    assert all(decision.label for decision in config.decisions.values())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("decisions.funnel", "failure_probability", 1.5),
        ("decisions.ads", "fixed_cost_eur", -1),
        ("levers.webinar", "invite_rate", -0.1),
        ("simulation", "volume_cv", 2.0),
    ],
)
def test_invalid_values_are_rejected(raw_config, section, key, value):
    broken = copy.deepcopy(raw_config)
    target = broken
    for part in section.split("."):
        target = target[part]
    target[key] = value
    with pytest.raises(ValueError, match=key):
        scenarios.parse_config(broken)


def test_missing_decision_is_rejected(raw_config):
    del raw_config["decisions"]["webinar"]
    with pytest.raises(ValueError, match="webinar"):
        scenarios.parse_config(raw_config)


def test_funnel_only_touches_funnel_columns(prepared):
    base = prepared.base
    treated = scenarios.apply_funnel(base)
    changed = [column for column in base.columns if not base[column].equals(treated[column])]
    assert set(changed) <= set(scenarios.FUNNEL_TREATMENT)


def test_ads_step_up_moves_paid_rows_one_level(prepared):
    base = prepared.base
    stepped = scenarios.apply_ads_step_up(base, prepared.ads_response)
    paid = base["channel"].isin(scenarios.PAID_CHANNELS)
    expected = base.loc[paid, "ad_budget_level"].map(scenarios.BUDGET_STEP_UP)
    assert stepped.loc[paid, "ad_budget_level"].equals(expected)
    assert base.loc[~paid].equals(stepped.loc[~paid])


def test_ads_mediators_follow_history(prepared):
    response = prepared.ads_response
    assert response.loc["saturated", "mean_lead_score"] < response.loc["medium", "mean_lead_score"]
    assert response.loc["saturated", "mean_cost_eur"] > response.loc["medium", "mean_cost_eur"]


def test_candidates_exclude_already_treated(prepared):
    base = prepared.base
    assert not (scenarios.webinar_candidates(base) & (base["webinar_invited"] == 1)).any()
    assert not (scenarios.new_product_candidates(base) & (base["new_product_offer"] == 1)).any()


def test_expected_values_are_row_aligned(prepared):
    baseline, values = scenarios.scenario_expected_values(
        prepared.model, prepared.base, prepared.config, prepared.ads_response
    )
    assert len(baseline) == len(prepared.base)
    assert set(values) == set(scenarios.DECISION_KEYS)
    assert all(len(array) == len(baseline) for array in values.values())


def test_zero_reach_levers_change_nothing(prepared):
    levers = copy.deepcopy(prepared.config.levers)
    levers["webinar"]["invite_rate"] = 0.0
    levers["new_product"]["offer_rate"] = 0.0
    config = dataclasses.replace(prepared.config, levers=levers)
    baseline, values = scenarios.scenario_expected_values(
        prepared.model, prepared.base, config, prepared.ads_response
    )
    np.testing.assert_allclose(values["webinar"], baseline)
    np.testing.assert_allclose(values["new_product"], baseline)
