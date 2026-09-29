"""Read-only analytical tools the agent can call.

The tools expose what a real analyst would have: history, model estimates with their
uncertainty, the simulated distributions and the validation checks. They deliberately
do NOT expose the synthetic ground truth (``true_*`` columns): the agent must reason
from estimates, as it would with real data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from montecarlo_decisions.artifacts import Artifacts
from montecarlo_decisions.scenarios import DECISION_KEYS

LEVER_KEYS = ("funnel", "webinar", "new_product")
ALL_CHANNELS = "all"
GROUND_TRUTH_CHECKS = frozenset({"counterfactual_recovery"})  # their detail quotes true values


class ToolInputError(ValueError):
    """Invalid arguments; reported back to the model as an error tool result."""


def _round(value: Any, digits: int = 2) -> Any:
    if isinstance(value, float | np.floating):
        return round(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _records(frame: pd.DataFrame, digits: int = 2) -> list[dict[str, Any]]:
    return [
        {key: _round(value, digits) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


def _object_schema(properties: dict[str, Any] | None = None) -> dict[str, Any]:
    properties = properties or {}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class Toolbox:
    """The agent's tools, bound to one set of artifacts."""

    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts
        self.labels = artifacts.config.labels()
        channels = sorted(artifacts.dataset["channel"].unique())
        self.specs = {
            spec.name: spec
            for spec in [
                ToolSpec(
                    "analyze_business",
                    "Historical baseline by acquisition channel: opportunities, conversion rate, "
                    "revenue, contribution profit, mean acquisition cost and mean lead score. "
                    "Use it first to understand the starting point.",
                    _object_schema(
                        {"channel": {"type": "string", "enum": [ALL_CHANNELS, *channels]}}
                    ),
                    self.analyze_business,
                ),
                ToolSpec(
                    "get_lever_uplift",
                    "Counterfactual uplift of a lever estimated by the ML models on the history: "
                    "conversion lift and expected value lift per opportunity, with a 95% "
                    "bootstrap interval. Not available for ads (see get_ads_budget_regimes).",
                    _object_schema({"lever": {"type": "string", "enum": list(LEVER_KEYS)}}),
                    self.get_lever_uplift,
                ),
                ToolSpec(
                    "get_ads_budget_regimes",
                    "Paid-media performance by historical budget level (low to saturated): "
                    "conversion, lead score, cost and contribution per opportunity. Evidence "
                    "of saturation for the 'double ad spend' decision.",
                    _object_schema(),
                    self.get_ads_budget_regimes,
                ),
                ToolSpec(
                    "get_decision_distribution",
                    "Monte Carlo distribution of the incremental profit of one decision over the "
                    "planning horizon, plus the execution-risk assumptions used to simulate it.",
                    _object_schema({"decision": {"type": "string", "enum": list(DECISION_KEYS)}}),
                    self.get_decision_distribution,
                ),
                ToolSpec(
                    "compare_decisions",
                    "Side-by-side ranking of all decisions by expected incremental profit, with "
                    "downside (P10, CVaR 5%), probability of loss and probability of being best.",
                    _object_schema(),
                    self.compare_decisions,
                ),
                ToolSpec(
                    "get_validation_checks",
                    "Quality gates of the analysis run (data integrity, model discrimination and "
                    "calibration, counterfactual recovery, Monte Carlo precision).",
                    _object_schema(),
                    self.get_validation_checks,
                ),
            ]
        }

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.definition() for spec in self.specs.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self.specs.get(name)
        if spec is None:
            raise ToolInputError(f"Unknown tool {name!r}. Available: {sorted(self.specs)}")
        expected = set(spec.input_schema["properties"])
        if set(arguments) != expected:
            raise ToolInputError(
                f"{name} expects arguments {sorted(expected)}, got {sorted(arguments)}"
            )
        for key, value in arguments.items():
            allowed = spec.input_schema["properties"][key].get("enum")
            if allowed is not None and value not in allowed:
                raise ToolInputError(f"{key}={value!r} is not one of {allowed}")
        return spec.handler(**arguments)

    def call_json(self, name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(self.call(name, arguments), ensure_ascii=False)

    # --- Tools ------------------------------------------------------------------------

    def analyze_business(self, channel: str) -> dict[str, Any]:
        data = self.artifacts.dataset
        if channel != ALL_CHANNELS:
            data = data[data["channel"] == channel]
        table = (
            data.groupby("channel")
            .agg(
                opportunities=("transaction_id", "count"),
                conversion_rate=("converted_to_sale", "mean"),
                revenue_eur=("revenue_eur", "sum"),
                contribution_profit_eur=("contribution_profit_eur", "sum"),
                mean_acquisition_cost_eur=("cost_attributed_eur", "mean"),
                mean_lead_score=("lead_score", "mean"),
            )
            .sort_values("contribution_profit_eur", ascending=False)
            .reset_index()
        )
        return {
            "period": [str(data["date"].min()), str(data["date"].max())],
            "channels": _records(table, 4),
        }

    def get_lever_uplift(self, lever: str) -> dict[str, Any]:
        row = self.artifacts.uplift.set_index("lever").loc[lever]
        return {
            "lever": lever,
            "label": row["label"],
            "description": row["description"],
            "sample_size": int(row["sample_size"]),
            "baseline_conversion": _round(row["baseline_conversion"], 4),
            "scenario_conversion": _round(row["scenario_conversion"], 4),
            "conversion_lift_pct": _round(row["conversion_lift_pct"], 4),
            "value_lift_per_opportunity_eur": _round(row["value_lift_per_opportunity_eur"]),
            "value_lift_95ci_eur": [
                _round(row["value_lift_ci_low_eur"]),
                _round(row["value_lift_ci_high_eur"]),
            ],
        }

    def get_ads_budget_regimes(self) -> dict[str, Any]:
        return {"regimes": _records(self.artifacts.ads_regimes, 4)}

    def get_decision_distribution(self, decision: str) -> dict[str, Any]:
        profit = self.artifacts.simulations.loc[
            self.artifacts.simulations["decision"] == decision, "incremental_profit_eur"
        ].to_numpy()
        row = self.artifacts.summary.set_index("decision").loc[decision]
        assumptions = self.artifacts.config.decisions[decision]
        return {
            "decision": decision,
            "label": self.labels[decision],
            "n_simulations": len(profit),
            "expected_profit_eur": _round(profit.mean()),
            "percentiles_eur": {
                f"p{q}": _round(np.percentile(profit, q)) for q in (5, 10, 25, 50, 75, 90, 95)
            },
            "cvar_5_eur": _round(row["cvar_5_eur"]),
            "probability_loss": _round(row["probability_loss"], 4),
            "probability_best": _round(row["probability_best"], 4),
            "expected_roi": _round(row["expected_roi"]),
            "breakeven_failure_probability": _round(row["breakeven_failure_probability"], 4),
            "assumptions": {
                "fixed_cost_eur": assumptions.fixed_cost_eur,
                "failure_probability": assumptions.failure_probability,
                "realization_mean": assumptions.realization_mean,
                "realization_sd": assumptions.realization_sd,
                "rationale": assumptions.rationale,
            },
        }

    def compare_decisions(self) -> dict[str, Any]:
        columns = [
            "ranking",
            "decision",
            "label",
            "expected_profit_eur",
            "p10_eur",
            "p90_eur",
            "cvar_5_eur",
            "probability_loss",
            "probability_best",
            "expected_roi",
        ]
        return {"ranking": _records(self.artifacts.summary[columns], 4)}

    def get_validation_checks(self) -> dict[str, Any]:
        """Check names and outcomes. Details that quote the synthetic truth are withheld."""
        checks = []
        for check in self.artifacts.checks:
            detail = check["detail"]
            if check["name"] in GROUND_TRUTH_CHECKS:
                detail = (
                    "Estimated lever effects were validated against a reference; details "
                    "withheld from the agent."
                )
            checks.append({"name": check["name"], "passed": check["passed"], "detail": detail})
        return {"checks": checks}


def _es(value: float) -> str:
    """Spanish thousands separator: 90330 -> '90.330'."""
    return f"{value:,.0f}".replace(",", ".")


def summarize_result(name: str, result: dict[str, Any]) -> str:
    """One-line, human-readable outcome of a tool call (for the dashboard trace)."""
    if name == "compare_decisions":
        ranking = result["ranking"]
        head = ranking[0]
        return (
            f"{head['label']} lidera con {_es(head['expected_profit_eur'])} EUR esperados "
            f"(P(perdida) {head['probability_loss']:.1%})."
        )
    if name == "get_decision_distribution":
        return (
            f"{result['label']}: E={_es(result['expected_profit_eur'])} EUR, "
            f"P10={_es(result['percentiles_eur']['p10'])}, "
            f"P90={_es(result['percentiles_eur']['p90'])}, "
            f"P(perdida) {result['probability_loss']:.1%}."
        )
    if name == "get_lever_uplift":
        low, high = result["value_lift_95ci_eur"]
        return (
            f"{result['label']}: conversion {result['conversion_lift_pct']:+.1%}, "
            f"valor {result['value_lift_per_opportunity_eur']:+.1f} EUR/oportunidad "
            f"[IC95 {low:.1f}, {high:.1f}]."
        )
    if name == "analyze_business":
        top = result["channels"][0]
        return (
            f"{top['channel']} aporta el mayor beneficio de contribucion "
            f"({_es(top['contribution_profit_eur'])} EUR)."
        )
    if name == "get_ads_budget_regimes":
        regimes = {row["ad_budget_level"]: row for row in result["regimes"]}
        parts = [
            f"{level}: {row['contribution_per_opportunity_eur']:.1f} EUR/op"
            for level, row in regimes.items()
        ]
        return "Contribucion por oportunidad paid: " + ", ".join(parts) + "."
    if name == "get_validation_checks":
        passed = sum(check["passed"] for check in result["checks"])
        return f"{passed}/{len(result['checks'])} comprobaciones superadas."
    return "Resultado disponible."
