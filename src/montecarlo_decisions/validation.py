"""Deterministic quality gates of a pipeline run.

They check that the analysis is *sound*, never which decision wins: the ranking is an
output of the case, not an acceptance criterion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MIN_OUT_OF_TIME_AUC = 0.68
MAX_CALIBRATION_ERROR = 0.03
RANKING_Z_THRESHOLD = 3.0


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def check_data_integrity(history: pd.DataFrame) -> Check:
    converted = history["converted_to_sale"] == 1
    problems = []
    if not history["transaction_id"].is_unique:
        problems.append("duplicated transaction_id")
    if not history["converted_to_sale"].isin([0, 1]).all():
        problems.append("converted_to_sale is not binary")
    if (history["cost_attributed_eur"] < 0).any():
        problems.append("negative acquisition cost")
    if (
        history.loc[converted, "aov_eur"].isna().any()
        or history.loc[~converted, "aov_eur"].notna().any()
    ):
        problems.append("aov_eur must be present exactly for converted rows")
    if (history["webinar_attended"] > history["webinar_invited"]).any():
        problems.append("webinar attendance without invitation")
    detail = (
        "; ".join(problems)
        or f"{len(history):,} rows, {history['date'].min()} to {history['date'].max()}"
    )
    return Check("data_integrity", not problems, detail)


def check_model_quality(model_report: dict) -> list[Check]:
    classification = model_report["classification"]
    auc = classification["auc"]
    ece = classification["expected_calibration_error"]
    return [
        Check(
            "conversion_model_discrimination",
            auc >= MIN_OUT_OF_TIME_AUC,
            f"out-of-time AUC {auc:.3f} (threshold {MIN_OUT_OF_TIME_AUC})",
        ),
        Check(
            "conversion_model_calibration",
            ece <= MAX_CALIBRATION_ERROR,
            f"expected calibration error {ece:.4f} (threshold {MAX_CALIBRATION_ERROR})",
        ),
    ]


def check_counterfactual_recovery(uplift: pd.DataFrame) -> Check:
    parts = [
        f"{row.lever}: est {row.value_lift_per_opportunity_eur:.1f} "
        f"[{row.value_lift_ci_low_eur:.1f}, {row.value_lift_ci_high_eur:.1f}] "
        f"vs true {row.true_value_lift_per_opportunity_eur:.1f}"
        for row in uplift.itertuples()
    ]
    return Check(
        "counterfactual_recovery", bool(uplift["truth_in_interval"].all()), "; ".join(parts)
    )


def check_ranking_resolved(simulations: pd.DataFrame, summary: pd.DataFrame) -> Check:
    """The gap between #1 and #2 is many Monte Carlo standard errors wide (paired)."""
    if len(summary) < 2:
        return Check("ranking_resolved", True, "single decision")
    first, second = summary["decision"].iloc[0], summary["decision"].iloc[1]
    wide = simulations.pivot(
        index="simulation", columns="decision", values="incremental_profit_eur"
    )
    diff = (wide[first] - wide[second]).to_numpy()
    standard_error = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.inf
    z = diff.mean() / standard_error if standard_error > 0 else np.inf
    return Check(
        "ranking_resolved",
        bool(z >= RANKING_Z_THRESHOLD),
        f"{first} vs {second}: paired gap {diff.mean():,.0f} EUR, z = {z:.1f} "
        f"(threshold {RANKING_Z_THRESHOLD})",
    )


def run_checks(
    history: pd.DataFrame,
    model_report: dict,
    uplift: pd.DataFrame,
    simulations: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[Check]:
    return [
        check_data_integrity(history),
        *check_model_quality(model_report),
        check_counterfactual_recovery(uplift),
        check_ranking_resolved(simulations, summary),
    ]
