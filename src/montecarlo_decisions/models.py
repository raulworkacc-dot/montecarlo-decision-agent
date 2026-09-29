"""Predictive models that turn the history into an expected value per opportunity.

    expected value = P(sale) * E[ticket | sale] * E[margin] - acquisition cost - P(sale) * support

Design choices (see docs/methodology.md for the reasoning):

* ``month_index`` is a feature. Funnel levers were rolled out over time while baseline
  conversion drifted upwards; without a time control the drift is attributed to the
  levers (confounding by calendar time).
* Acquisition cost and daily spend are *not* conversion features: they do not cause a
  sale, and keeping them would make a cost change in a scenario move the predicted
  conversion.
* The ticket model (gradient boosting with native categorical splits) is fit on
  log(ticket) and back-transformed with Duan's smearing estimator, because
  exp(E[log y]) underestimates E[y].
* Evaluation uses a temporal holdout (the most recent 20% of opportunities), which is
  closer to how the model is used, i.e. to extrapolate to the next months.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from montecarlo_decisions.config import SEED
from montecarlo_decisions.synthetic import SUPPORT_COST_EUR, month_index

CATEGORICAL_FEATURES = [
    "channel",
    "campaign_objective",
    "customer_segment",
    "lifecycle_stage",
    "device",
    "geo_region",
    "ad_budget_level",
    "landing_variant",
    "cta_variant",
]
NUMERIC_FEATURES = [
    "lead_magnet",
    "checkout_simplified",
    "webinar_invited",
    "webinar_attended",
    "new_product_offer",
    "lead_score",
    "month_index",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
NAIVE_FEATURES = [feature for feature in FEATURES if feature != "month_index"]
MARGIN_FEATURES = ["channel", "new_product_offer"]
HOLDOUT_FRACTION = 0.20


def with_time_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``month_index`` (months since the start of the history) if it is missing."""
    if "month_index" in frame.columns:
        return frame
    return frame.assign(month_index=month_index(frame["date"]))


def _preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )


def _split_features(features: list[str]) -> tuple[list[str], list[str]]:
    return (
        [f for f in features if f in CATEGORICAL_FEATURES],
        [f for f in features if f not in CATEGORICAL_FEATURES],
    )


def fit_conversion_model(frame: pd.DataFrame, features: list[str] = FEATURES) -> Pipeline:
    model = Pipeline(
        [
            ("preprocess", _preprocessor(*_split_features(features))),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )
    return model.fit(frame[features], frame["converted_to_sale"])


@dataclass
class TicketModel:
    """Gradient boosting on log(ticket) with a Duan smearing back-transformation."""

    pipeline: Pipeline
    smearing: float
    features: list[str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.pipeline.predict(frame[self.features])) * self.smearing


def fit_ticket_model(
    frame: pd.DataFrame, seed: int = SEED, features: list[str] = FEATURES
) -> TicketModel:
    sold = frame[frame["converted_to_sale"] == 1]
    log_ticket = np.log(sold["aov_eur"].to_numpy())
    categorical, numeric = _split_features(features)
    encoder = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", encoder),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.05,
                    min_samples_leaf=30,
                    categorical_features=list(range(len(categorical))),
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(sold[features], log_ticket)
    residuals = log_ticket - pipeline.predict(sold[features])
    return TicketModel(pipeline, float(np.mean(np.exp(residuals))), list(features))


def fit_margin_model(frame: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        [
            ("preprocess", _preprocessor(["channel"], ["new_product_offer"])),
            ("regressor", LinearRegression()),
        ]
    )
    return model.fit(frame[MARGIN_FEATURES], frame["gross_margin_pct"])


@dataclass
class ValueComponents:
    conversion: np.ndarray
    ticket: np.ndarray
    margin: np.ndarray
    cost: np.ndarray

    @property
    def expected_value(self) -> np.ndarray:
        gross = self.conversion * self.ticket * self.margin
        return gross - self.cost - self.conversion * SUPPORT_COST_EUR


@dataclass
class ValueModel:
    """The three fitted models combined into an expected value per opportunity."""

    conversion: Pipeline
    ticket: TicketModel
    margin: Pipeline
    features: list[str]

    def components(self, frame: pd.DataFrame) -> ValueComponents:
        frame = with_time_index(frame)
        return ValueComponents(
            conversion=self.conversion.predict_proba(frame[self.features])[:, 1],
            ticket=self.ticket.predict(frame),
            margin=self.margin.predict(frame[MARGIN_FEATURES]),
            cost=frame["cost_attributed_eur"].to_numpy(dtype=float),
        )

    def expected_value(self, frame: pd.DataFrame) -> np.ndarray:
        return self.components(frame).expected_value


def fit_value_model(frame: pd.DataFrame, seed: int = SEED, time_control: bool = True) -> ValueModel:
    """Fit the three models. ``time_control=False`` reproduces the naive specification."""
    features = FEATURES if time_control else NAIVE_FEATURES
    frame = with_time_index(frame)
    return ValueModel(
        conversion=fit_conversion_model(frame, features),
        ticket=fit_ticket_model(frame, seed, features),
        margin=fit_margin_model(frame),
        features=list(features),
    )


def bootstrap_rows(n_rows: int, n_replicates: int, seed: int = SEED) -> list[np.ndarray]:
    """Row indices of each bootstrap resample, drawn up front so results do not depend
    on how the replicates are scheduled across workers."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_rows, n_rows) for _ in range(n_replicates)]


def temporal_split(
    frame: pd.DataFrame, holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the oldest opportunities, validate on the most recent ones."""
    ordered = frame.sort_values("date", kind="stable")
    cut = round(len(ordered) * (1 - holdout_fraction))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def gain_chart(actual: np.ndarray, score: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Cumulative share of conversions captured by score decile (best decile first)."""
    order = np.argsort(-score, kind="stable")
    actual, score = np.asarray(actual)[order], np.asarray(score)[order]
    bins = np.array_split(np.arange(len(actual)), n_bins)
    total = max(1, int(actual.sum()))
    rows, captured = [], 0
    for decile, index in enumerate(bins, start=1):
        captured += int(actual[index].sum())
        rows.append(
            {
                "decile": decile,
                "avg_score": float(score[index].mean()),
                "conversion_rate": float(actual[index].mean()),
                "capture_pct": captured / total,
                "population_pct": (index[-1] + 1) / len(actual),
            }
        )
    return rows


def calibration_table(
    actual: np.ndarray, score: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Observed vs predicted conversion by predicted-probability decile."""
    order = np.argsort(score, kind="stable")
    bins = np.array_split(order, n_bins)
    return [
        {
            "bin": i,
            "predicted": float(np.mean(score[index])),
            "observed": float(np.mean(actual[index])),
            "count": len(index),
        }
        for i, index in enumerate(bins, start=1)
    ]


def expected_calibration_error(calibration: list[dict[str, float]]) -> float:
    total = sum(row["count"] for row in calibration)
    return float(
        sum(row["count"] * abs(row["predicted"] - row["observed"]) for row in calibration) / total
    )


def _residual_bands(residuals: np.ndarray) -> list[dict[str, object]]:
    bands = [
        ("Sobreestima > 300 EUR", residuals < -300),
        ("Sobreestima 100-300 EUR", (residuals >= -300) & (residuals < -100)),
        ("Ajuste fino +/- 100 EUR", np.abs(residuals) <= 100),
        ("Infraestima 100-300 EUR", (residuals > 100) & (residuals <= 300)),
        ("Infraestima > 300 EUR", residuals > 300),
    ]
    return [{"label": label, "count": int(mask.sum())} for label, mask in bands]


def evaluate_models(frame: pd.DataFrame, seed: int = SEED) -> dict[str, object]:
    """Out-of-time metrics for the conversion and ticket models."""
    frame = with_time_index(frame)
    train, test = temporal_split(frame)
    model = fit_value_model(train, seed)
    score = model.conversion.predict_proba(test[FEATURES])[:, 1]
    actual = test["converted_to_sale"].to_numpy()
    calibration = calibration_table(actual, score)

    sold_test = test[test["converted_to_sale"] == 1]
    predicted_ticket = model.ticket.predict(sold_test)
    residuals = sold_test["aov_eur"].to_numpy() - predicted_ticket
    return {
        "split": {
            "strategy": "temporal",
            "train_rows": len(train),
            "test_rows": len(test),
            "test_period_start": str(test["date"].min()),
            "test_period_end": str(test["date"].max()),
        },
        "classification": {
            "name": "Regresion logistica",
            "auc": float(roc_auc_score(actual, score)),
            "brier": float(brier_score_loss(actual, score)),
            "expected_calibration_error": expected_calibration_error(calibration),
            "positive_rate": float(actual.mean()),
            "avg_predicted_prob": float(score.mean()),
            "gain_chart": gain_chart(actual, score),
            "calibration": calibration,
        },
        "regression": {
            "name": "Gradient boosting (log-ticket + smearing)",
            "mae_eur": float(mean_absolute_error(sold_test["aov_eur"], predicted_ticket)),
            "r2": float(r2_score(sold_test["aov_eur"], predicted_ticket)),
            "mean_residual_eur": float(residuals.mean()),
            "p90_abs_error_eur": float(np.percentile(np.abs(residuals), 90)),
            "smearing_factor": model.ticket.smearing,
            "residual_bands": _residual_bands(residuals),
        },
    }
