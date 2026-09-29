"""Decision scenarios: business assumptions and the interventions each decision applies.

Every intervention returns a *row-aligned* expected value: element ``i`` is the expected
contribution of opportunity ``i`` if the decision were taken. Keeping rows aligned with
the baseline lets the simulation compare decisions on the same opportunities (common
random numbers), so the ranking reflects the decisions and not sampling noise.

Stochastic rollouts (only some leads get invited, only some attend) are handled by
taking the expectation over the rollout, not by sampling it: the mix is known, so
sampling it would only add avoidable noise.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from montecarlo_decisions.config import SCENARIOS_PATH
from montecarlo_decisions.models import ValueModel
from montecarlo_decisions.synthetic import NEW_PRODUCT_SEGMENTS, PAID_CHANNELS, WARM_CHANNELS

DECISION_KEYS = ("funnel", "webinar", "ads", "new_product")
BUDGET_STEP_UP = {"low": "medium", "medium": "high", "high": "saturated", "saturated": "saturated"}
FUNNEL_TREATMENT = {
    "landing_variant": "landing_v2",
    "cta_variant": "benefit_cta",
    "lead_magnet": 1,
    "checkout_simplified": 1,
}


@dataclass(frozen=True)
class Decision:
    key: str
    label: str
    fixed_cost_eur: float
    failure_probability: float
    realization_mean: float
    realization_sd: float
    cost_overrun_sd: float
    rationale: str


@dataclass(frozen=True)
class SimulationSettings:
    horizon_months: int
    base_window_start: str
    volume_cv: float
    n_bootstrap_models: int
    chunk_size: int


@dataclass(frozen=True)
class ScenarioConfig:
    simulation: SimulationSettings
    decisions: dict[str, Decision]
    levers: dict[str, dict[str, float]]

    def labels(self) -> dict[str, str]:
        return {key: decision.label for key, decision in self.decisions.items()}


def _require_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{name}={value} must be within [{low}, {high}]")


def parse_config(raw: Mapping[str, object]) -> ScenarioConfig:
    """Build and validate a config from a parsed TOML mapping."""
    simulation = SimulationSettings(**raw["simulation"])
    _require_range("simulation.horizon_months", simulation.horizon_months, 1, 60)
    _require_range("simulation.volume_cv", simulation.volume_cv, 0.0, 1.0)
    _require_range("simulation.n_bootstrap_models", simulation.n_bootstrap_models, 1, 500)
    _require_range("simulation.chunk_size", simulation.chunk_size, 1, 100_000)

    decisions_raw = raw["decisions"]
    missing = set(DECISION_KEYS) - set(decisions_raw)
    if missing:
        raise ValueError(f"Missing decisions in config: {sorted(missing)}")
    decisions = {}
    for key in DECISION_KEYS:
        decision = Decision(key=key, **decisions_raw[key])
        _require_range(f"{key}.fixed_cost_eur", decision.fixed_cost_eur, 1e-9, 1e9)
        _require_range(f"{key}.failure_probability", decision.failure_probability, 0.0, 1.0)
        _require_range(f"{key}.realization_mean", decision.realization_mean, 1e-9, 10.0)
        _require_range(f"{key}.realization_sd", decision.realization_sd, 0.0, 5.0)
        _require_range(f"{key}.cost_overrun_sd", decision.cost_overrun_sd, 0.0, 5.0)
        decisions[key] = decision

    levers = {name: dict(values) for name, values in raw["levers"].items()}
    for name, key in [
        ("webinar", "invite_rate"),
        ("webinar", "attendance_rate"),
        ("new_product", "offer_rate"),
    ]:
        _require_range(f"levers.{name}.{key}", levers[name][key], 0.0, 1.0)
    _require_range("levers.ads.volume_uplift", levers["ads"]["volume_uplift"], 0.0, 10.0)
    return ScenarioConfig(simulation, decisions, levers)


def load_config(path: Path = SCENARIOS_PATH) -> ScenarioConfig:
    with Path(path).open("rb") as handle:
        return parse_config(tomllib.load(handle))


# --- Data-estimated response of paid media ---------------------------------------------


def estimate_ads_response(history: pd.DataFrame) -> pd.DataFrame:
    """Mean lead score and cost per opportunity of paid channels at each budget level.

    Used as the *mediator* adjustment of the ads scenario: when budget moves one level up,
    the lead score and the acquisition cost of paid leads move to what the history shows
    at that level.
    """
    paid = history[history["channel"].isin(PAID_CHANNELS)]
    response = paid.groupby("ad_budget_level").agg(
        opportunities=("transaction_id", "count"),
        conversion_rate=("converted_to_sale", "mean"),
        mean_lead_score=("lead_score", "mean"),
        mean_cost_eur=("cost_attributed_eur", "mean"),
        contribution_per_opportunity_eur=("contribution_profit_eur", "mean"),
    )
    order = [level for level in BUDGET_STEP_UP if level in response.index]
    return response.loc[order]


# --- Interventions ----------------------------------------------------------------------


def apply_funnel(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(**FUNNEL_TREATMENT)


def webinar_candidates(frame: pd.DataFrame) -> np.ndarray:
    warm = (frame["lifecycle_stage"] != "new_visitor") | frame["channel"].isin(WARM_CHANNELS)
    return (warm & (frame["webinar_invited"] == 0)).to_numpy()


def new_product_candidates(frame: pd.DataFrame) -> np.ndarray:
    eligible = frame["customer_segment"].isin(NEW_PRODUCT_SEGMENTS)
    return (eligible & (frame["new_product_offer"] == 0)).to_numpy()


def apply_new_product(frame: pd.DataFrame, extra_cost_eur: float) -> pd.DataFrame:
    return frame.assign(
        new_product_offer=1, cost_attributed_eur=frame["cost_attributed_eur"] + extra_cost_eur
    )


def apply_ads_step_up(frame: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    """Move every paid opportunity one budget level up, with its mediators."""
    paid = frame["channel"].isin(PAID_CHANNELS)
    current = frame["ad_budget_level"]
    target = current.map(BUDGET_STEP_UP).where(paid, current)
    score_shift = target.map(response["mean_lead_score"]) - current.map(response["mean_lead_score"])
    cost_ratio = target.map(response["mean_cost_eur"]) / current.map(response["mean_cost_eur"])
    return frame.assign(
        ad_budget_level=target,
        lead_score=np.clip(
            (frame["lead_score"] + score_shift.where(paid, 0.0)).round(), 1, 99
        ).astype(int),
        cost_attributed_eur=frame["cost_attributed_eur"] * cost_ratio.where(paid, 1.0),
    )


def scenario_expected_values(
    model: ValueModel,
    base: pd.DataFrame,
    config: ScenarioConfig,
    ads_response: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Baseline and per-decision expected value for each opportunity in ``base``."""
    ev = model.expected_value
    baseline = ev(base)
    levers = config.levers

    webinar = levers["webinar"]
    invited_only = ev(base.assign(webinar_invited=1, webinar_attended=0))
    attended = ev(base.assign(webinar_invited=1, webinar_attended=1))
    webinar_treated = (
        webinar["attendance_rate"] * attended + (1 - webinar["attendance_rate"]) * invited_only
    )
    webinar_share = webinar["invite_rate"] * webinar_candidates(base)

    product = levers["new_product"]
    offered = ev(apply_new_product(base, product["extra_cost_per_offer_eur"]))
    product_share = product["offer_rate"] * new_product_candidates(base)

    paid = base["channel"].isin(PAID_CHANNELS).to_numpy()
    stepped_up = ev(apply_ads_step_up(base, ads_response))
    ads_value = np.where(paid, (1 + levers["ads"]["volume_uplift"]) * stepped_up, baseline)

    return baseline, {
        "funnel": ev(apply_funnel(base)),
        "webinar": baseline + webinar_share * (webinar_treated - baseline),
        "ads": ads_value,
        "new_product": baseline + product_share * (offered - baseline),
    }
