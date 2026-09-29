"""Counterfactual uplift of each lever, validated against the synthetic ground truth.

For each lever we take the opportunities it could apply to, copy them, force the
treatment and re-predict with the models. Three estimates are reported side by side:

* ``model``: the time-controlled models used by the simulation;
* ``naive``: the same models without ``month_index`` (the original specification);
* ``true``: the structural DGP functions, i.e. the real causal effect.

The gap between ``naive`` and ``true`` quantifies the confounding bias; the gap between
``model`` and ``true`` is the error that remains after controlling for time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from montecarlo_decisions.config import SEED
from montecarlo_decisions.models import ValueModel
from montecarlo_decisions.scenarios import (
    apply_funnel,
    apply_new_product,
    new_product_candidates,
    webinar_candidates,
)
from montecarlo_decisions.synthetic import true_conversion_probability, true_expected_value

INTERVAL_MAX_ROWS = 5_000


@dataclass(frozen=True)
class Lever:
    key: str
    label: str
    description: str
    population: Callable[[pd.DataFrame], np.ndarray]
    treat: Callable[[pd.DataFrame], pd.DataFrame]


def levers(extra_cost_per_offer_eur: float) -> list[Lever]:
    return [
        Lever(
            "funnel",
            "Optimizacion integral del funnel",
            "Landing v2, CTA de beneficio, lead magnet y checkout simplificado a la vez, "
            "aplicados a todas las oportunidades.",
            lambda frame: np.ones(len(frame), dtype=bool),
            apply_funnel,
        ),
        Lever(
            "webinar",
            "Asistencia a webinar comercial",
            "Leads templados que aun no habian sido invitados pasan a asistir a un webinar.",
            webinar_candidates,
            lambda frame: frame.assign(webinar_invited=1, webinar_attended=1),
        ),
        Lever(
            "new_product",
            "Oferta de nuevo producto",
            "Segmentos elegibles reciben la oferta de mayor ticket (menor conversion, "
            "mayor valor por venta).",
            new_product_candidates,
            lambda frame: apply_new_product(frame, extra_cost_per_offer_eur),
        ),
    ]


def _lift(
    base: pd.DataFrame,
    treated: pd.DataFrame,
    conversion: Callable[[pd.DataFrame], np.ndarray],
    value: Callable[[pd.DataFrame], np.ndarray],
) -> tuple[float, float, float, float]:
    p0, p1 = conversion(base).mean(), conversion(treated).mean()
    value_lift = float((value(treated) - value(base)).mean())
    return float(p0), float(p1), float(p1 / p0 - 1), value_lift


def _model_conversion(model: ValueModel) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda frame: model.components(frame).conversion


def lever_effects(
    history: pd.DataFrame,
    model: ValueModel,
    naive_model: ValueModel,
    extra_cost_per_offer_eur: float,
) -> pd.DataFrame:
    """One row per lever with model, naive and true conversion/value lift."""
    rows = []
    for lever in levers(extra_cost_per_offer_eur):
        base = history[lever.population(history)]
        treated = lever.treat(base)
        p0, p1, lift, value_lift = _lift(
            base, treated, _model_conversion(model), model.expected_value
        )
        _, _, naive_lift, naive_value = _lift(
            base, treated, _model_conversion(naive_model), naive_model.expected_value
        )
        _, _, true_lift, true_value = _lift(
            base, treated, true_conversion_probability, true_expected_value
        )
        rows.append(
            {
                "lever": lever.key,
                "label": lever.label,
                "description": lever.description,
                "sample_size": len(base),
                "observed_conversion": float(base["converted_to_sale"].mean()),
                "baseline_conversion": p0,
                "scenario_conversion": p1,
                "conversion_lift_pct": lift,
                "value_lift_per_opportunity_eur": value_lift,
                "naive_conversion_lift_pct": naive_lift,
                "naive_value_lift_per_opportunity_eur": naive_value,
                "true_conversion_lift_pct": true_lift,
                "true_value_lift_per_opportunity_eur": true_value,
            }
        )
    effects = pd.DataFrame(rows)
    effects["value_lift_error_pct"] = (
        effects["value_lift_per_opportunity_eur"] / effects["true_value_lift_per_opportunity_eur"]
        - 1
    )
    effects["naive_value_lift_error_pct"] = (
        effects["naive_value_lift_per_opportunity_eur"]
        / effects["true_value_lift_per_opportunity_eur"]
        - 1
    )
    return effects


@dataclass(frozen=True)
class LeverSample:
    """A lever's population (subsampled) and its treated copy, for replicate evaluation."""

    key: str
    base: pd.DataFrame
    treated: pd.DataFrame


def lever_samples(
    history: pd.DataFrame,
    extra_cost_per_offer_eur: float,
    max_rows: int = INTERVAL_MAX_ROWS,
    seed: int = SEED,
) -> list[LeverSample]:
    """Fixed subsample of at most ``max_rows`` opportunities per lever.

    The subsampling error of a mean lift is far below the width of the bootstrap
    interval, and it makes evaluating many model replicates cheap.
    """
    samples = []
    for lever in levers(extra_cost_per_offer_eur):
        base = history[lever.population(history)]
        if len(base) > max_rows:
            base = base.sample(max_rows, random_state=seed)
        samples.append(LeverSample(lever.key, base, lever.treat(base)))
    return samples


def value_lifts(model: ValueModel, samples: list[LeverSample]) -> np.ndarray:
    """Mean value lift of each lever under one model."""
    return np.array(
        [
            float((model.expected_value(s.treated) - model.expected_value(s.base)).mean())
            for s in samples
        ]
    )


def add_bootstrap_intervals(
    effects: pd.DataFrame, replicate_lifts: np.ndarray, level: float = 0.95
) -> pd.DataFrame:
    """Percentile interval of the value lift across bootstrap model refits.

    ``replicate_lifts`` has shape (n_replicates, n_levers), levers in ``effects`` order.
    ``truth_in_interval`` is the honest recovery test: an estimate can be off and still be
    fine if its stated uncertainty covers the truth; it is a problem when it does not.
    """
    tail = (1 - level) / 2 * 100
    effects = effects.copy()
    effects["value_lift_ci_low_eur"] = np.percentile(replicate_lifts, tail, axis=0)
    effects["value_lift_ci_high_eur"] = np.percentile(replicate_lifts, 100 - tail, axis=0)
    effects["truth_in_interval"] = effects["true_value_lift_per_opportunity_eur"].between(
        effects["value_lift_ci_low_eur"], effects["value_lift_ci_high_eur"]
    )
    return effects
