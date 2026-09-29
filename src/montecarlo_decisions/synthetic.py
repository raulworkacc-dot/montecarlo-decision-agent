"""Synthetic data-generating process (DGP) for a digital-marketing sales funnel.

Each row is one commercial opportunity (Jan 2024 - Apr 2026) with its acquisition
context, the marketing levers active at the time and its economic outcome.

The DGP is deliberately *structural*: covariates are sampled first, then outcomes are
drawn from explicit functions of those covariates. The same functions are exposed as
``true_*`` helpers so the rest of the project can check its estimates against the
ground truth, something that is impossible with real data and is the main reason to
use a synthetic case at all.

Two properties of the DGP matter for the analysis:

* **Time confounding.** Funnel improvements (landing v2, benefit CTA, lead magnet,
  simplified checkout) roll out over time, while baseline conversion also drifts up
  month after month. A model without a time control attributes part of the drift to
  the funnel levers.
* **Mediation through lead quality.** Higher paid-media budget levels lower the lead
  score (saturation), so the effect of ad spend flows partly through ``lead_score``.

This module is the "world" of the case study. It is listed in
``.claude/protected_paths.json``: the agent may not tune the world to fit a conclusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from montecarlo_decisions.config import SEED

N_ROWS = 20_000
START_DATE = "2024-01-01"
END_DATE = "2026-04-30"
SUPPORT_COST_EUR = 22.0
AOV_LOG_SD = 0.23
MARGIN_SD = 0.035
CONVERSION_CLIP = (0.005, 0.58)
AOV_CLIP = (120.0, 5200.0)
MARGIN_CLIP = (0.52, 0.86)

CHANNELS = (
    "Facebook Ads",
    "Google Ads",
    "YouTube Organico",
    "SEO / Blog",
    "Email",
    "Afiliados",
    "Webinar",
)
PAID_CHANNELS = ("Facebook Ads", "Google Ads")
WARM_CHANNELS = ("Email", "Webinar", "SEO / Blog")
SEGMENTS = ("Creator", "Ecommerce", "B2B Services", "SMB", "Enterprise")
NEW_PRODUCT_SEGMENTS = ("Enterprise", "B2B Services", "Ecommerce")
DEVICES = ("desktop", "mobile", "tablet")
REGIONS = ("ES", "MX", "CO", "US Hispanic", "AR", "CL")
LIFECYCLE_STAGES = ("new_visitor", "known_lead", "returning_customer")
OBJECTIVES = ("cold_acquisition", "retargeting", "nurture", "launch", "evergreen")
BUDGET_LEVELS = ("organic_or_owned", "low", "medium", "high", "saturated")

# --- Structural coefficients of the conversion logit ---------------------------------
INTERCEPT_LOGIT = -3.15
LEAD_SCORE_LOGIT = 0.032  # per lead-score point above 50
MONTHLY_TREND_LOGIT = 0.004  # per month since START_DATE
CHANNEL_LOGIT = {
    "Email": 0.34,
    "Webinar": 0.46,
    "SEO / Blog": 0.16,
    "YouTube Organico": 0.08,
    "Google Ads": -0.03,
    "Afiliados": -0.08,
    "Facebook Ads": -0.16,
}
SEGMENT_LOGIT = {
    "Creator": -0.04,
    "Ecommerce": 0.08,
    "B2B Services": 0.12,
    "SMB": -0.10,
    "Enterprise": 0.04,
}
LIFECYCLE_LOGIT = {"new_visitor": -0.16, "known_lead": 0.22, "returning_customer": 0.38}
OBJECTIVE_LOGIT = {
    "cold_acquisition": -0.18,
    "retargeting": 0.16,
    "nurture": 0.22,
    "launch": -0.04,
    "evergreen": 0.03,
}
BUDGET_LOGIT = {
    "organic_or_owned": 0.0,
    "low": 0.06,
    "medium": 0.0,
    "high": -0.15,
    "saturated": -0.48,
}
LANDING_V2_LOGIT = 0.19
BENEFIT_CTA_LOGIT = 0.18
LEAD_MAGNET_LOGIT = 0.21
CHECKOUT_SIMPLIFIED_LOGIT = 0.16
WEBINAR_ATTENDED_LOGIT = 0.48
WEBINAR_INVITED_ONLY_LOGIT = 0.04
NEW_PRODUCT_LOGIT = -0.22

# --- Lead quality (lead_score) --------------------------------------------------------
CHANNEL_QUALITY = {
    "Email": 14,
    "Webinar": 12,
    "SEO / Blog": 8,
    "YouTube Organico": 5,
    "Google Ads": 1,
    "Afiliados": -1,
    "Facebook Ads": -4,
}
SEGMENT_QUALITY = {"Enterprise": 8, "B2B Services": 6, "Ecommerce": 3, "Creator": 0, "SMB": -2}
LIFECYCLE_QUALITY = {"new_visitor": -6, "known_lead": 5, "returning_customer": 12}
BUDGET_QUALITY = {"organic_or_owned": 0, "low": 3, "medium": 0, "high": -6, "saturated": -15}

# --- Ticket and margin ------------------------------------------------------------------
BASE_AOV = {"Creator": 760, "Ecommerce": 920, "B2B Services": 1120, "SMB": 640, "Enterprise": 1850}
CHANNEL_AOV = {
    "Email": 1.03,
    "Webinar": 1.12,
    "SEO / Blog": 1.00,
    "YouTube Organico": 0.98,
    "Google Ads": 0.96,
    "Afiliados": 0.92,
    "Facebook Ads": 0.93,
}
NEW_PRODUCT_AOV_MULTIPLIER = 2.35
WEBINAR_AOV_MULTIPLIER = 1.08
BASE_MARGIN = 0.73
CHANNEL_MARGIN = {
    "Email": 0.04,
    "Webinar": 0.03,
    "SEO / Blog": 0.02,
    "YouTube Organico": 0.03,
    "Google Ads": -0.01,
    "Afiliados": -0.04,
    "Facebook Ads": -0.02,
}
NEW_PRODUCT_MARGIN_DELTA = -0.03

# --- Acquisition costs ------------------------------------------------------------------
BASE_DAILY_SPEND = {"Facebook Ads": 420.0, "Google Ads": 310.0}
BUDGET_SPEND_MULTIPLIER = {"low": 0.70, "medium": 1.00, "high": 1.65, "saturated": 2.55}
BASE_COST_PER_OPPORTUNITY = {
    "Facebook Ads": 12.5,
    "Google Ads": 10.5,
    "Afiliados": 7.0,
    "Webinar": 5.5,
    "SEO / Blog": 1.8,
    "YouTube Organico": 1.2,
    "Email": 0.7,
}
BUDGET_COST_MULTIPLIER = {
    "organic_or_owned": 1.0,
    "low": 0.88,
    "medium": 1.00,
    "high": 1.28,
    "saturated": 1.82,
}

# Channel mix changes over time: (period start, probabilities in CHANNELS order).
CHANNEL_MIX = (
    ("2024-01-01", (0.21, 0.15, 0.13, 0.17, 0.18, 0.10, 0.06)),
    ("2024-09-01", (0.24, 0.16, 0.12, 0.15, 0.16, 0.09, 0.08)),
    ("2025-04-01", (0.31, 0.19, 0.09, 0.12, 0.13, 0.08, 0.08)),
    ("2025-07-01", (0.23, 0.16, 0.12, 0.15, 0.17, 0.08, 0.09)),
)
# Paid-media budget level by month (inclusive upper bound YYYYMM -> level).
BUDGET_SCHEDULE = (
    (202404, "low"),
    (202408, "medium"),
    (202412, "high"),
    (202503, "medium"),
    (202506, "saturated"),
    (202509, "medium"),
    (202512, "high"),
    (202602, "saturated"),
    (999999, "medium"),
)


def _lookup(values: pd.Series | np.ndarray, table: Mapping[str, float]) -> np.ndarray:
    mapped = pd.Series(np.asarray(values)).map(table)
    if mapped.isna().any():
        unknown = sorted(set(pd.Series(np.asarray(values))[mapped.isna()]))
        raise KeyError(f"Values without a structural coefficient: {unknown}")
    return mapped.to_numpy(dtype=float)


def _draw(
    rng: np.random.Generator, values: Sequence[str], probs: np.ndarray, size: int
) -> np.ndarray:
    """Draw one category per row; ``probs`` is (n_values,) or (size, n_values)."""
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    cumulative = np.broadcast_to(probs.cumsum(axis=-1), (size, probs.shape[-1]))
    u = rng.random(size)
    index = np.minimum((cumulative < u[:, None]).sum(axis=1), len(values) - 1)
    return np.asarray(values, dtype=object)[index]


def month_index(dates: pd.Series) -> np.ndarray:
    """Months elapsed since START_DATE (0 for January 2024)."""
    parsed = pd.to_datetime(dates)
    start = pd.Timestamp(START_DATE)
    return ((parsed.dt.year - start.year) * 12 + parsed.dt.month - start.month).to_numpy()


def _sample_dates(rng: np.random.Generator, n_rows: int) -> pd.DatetimeIndex:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    t = np.arange(len(dates))
    trend = 1 + 0.0009 * t
    seasonality = 1 + 0.10 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
    launch_bump = np.where((dates >= "2025-10-01") & (dates <= "2025-12-15"), 1.16, 1.0)
    weights = trend * seasonality * launch_bump
    sampled = rng.choice(dates.to_numpy(), size=n_rows, replace=True, p=weights / weights.sum())
    return pd.DatetimeIndex(np.sort(sampled))


def _budget_level(dates: pd.DatetimeIndex, channel: np.ndarray) -> np.ndarray:
    year_month = dates.year * 100 + dates.month
    bounds = np.array([bound for bound, _ in BUDGET_SCHEDULE])
    levels = np.array([level for _, level in BUDGET_SCHEDULE], dtype=object)
    paid_level = levels[np.searchsorted(bounds, year_month.to_numpy(), side="left")]
    return np.where(np.isin(channel, PAID_CHANNELS), paid_level, "organic_or_owned")


def _rollout(
    rng: np.random.Generator, dates: pd.DatetimeIndex, schedule: Sequence[tuple[str, float]]
) -> np.ndarray:
    """Binary adoption flag whose probability changes at each (start date, share) step."""
    share = np.zeros(len(dates))
    for start, probability in schedule:
        share = np.where(dates >= pd.Timestamp(start), probability, share)
    return (rng.random(len(dates)) < share).astype(int)


def sample_covariates(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Everything known about an opportunity before its outcome is realised."""
    rng = np.random.default_rng(seed)
    dates = _sample_dates(rng, n_rows)

    period = np.searchsorted(
        pd.DatetimeIndex([start for start, _ in CHANNEL_MIX]), dates, side="right"
    )
    channel_probs = np.array([probs for _, probs in CHANNEL_MIX])[period - 1]
    channel = _draw(rng, CHANNELS, channel_probs, n_rows)
    segment = _draw(rng, SEGMENTS, np.array([0.20, 0.24, 0.24, 0.22, 0.10]), n_rows)
    device = _draw(rng, DEVICES, np.array([0.58, 0.35, 0.07]), n_rows)
    region = _draw(rng, REGIONS, np.array([0.46, 0.18, 0.10, 0.08, 0.10, 0.08]), n_rows)
    lifecycle = _draw(rng, LIFECYCLE_STAGES, np.array([0.54, 0.34, 0.12]), n_rows)

    paid = np.isin(channel, PAID_CHANNELS)
    owned = np.isin(channel, ("Email", "Webinar"))
    objective_probs = np.where(
        paid[:, None],
        [0.48, 0.26, 0.08, 0.12, 0.06],
        np.where(owned[:, None], [0.03, 0.18, 0.50, 0.12, 0.17], [0.22, 0.12, 0.18, 0.08, 0.40]),
    )
    objective = _draw(rng, OBJECTIVES, objective_probs, n_rows)

    budget = _budget_level(dates, channel)
    base_spend = pd.Series(channel).map(BASE_DAILY_SPEND).fillna(0.0).to_numpy()
    spend_multiplier = pd.Series(budget).map(BUDGET_SPEND_MULTIPLIER).fillna(0.0).to_numpy()
    spend = (
        np.maximum(0.0, rng.normal(base_spend * spend_multiplier, 35 + base_spend * 0.08)) * paid
    )
    cost = np.maximum(
        0.15,
        rng.lognormal(
            np.log(
                _lookup(channel, BASE_COST_PER_OPPORTUNITY)
                * _lookup(budget, BUDGET_COST_MULTIPLIER)
            ),
            0.22,
        ),
    )

    landing_v2 = _rollout(rng, dates, [("2024-09-15", 0.42), ("2025-01-01", 0.70)])
    benefit_cta = _rollout(rng, dates, [("2025-03-15", 0.48), ("2025-07-01", 0.72)])
    lead_magnet = _rollout(rng, dates, [("2025-03-15", 0.42), ("2025-07-01", 0.76)])
    checkout = _rollout(rng, dates, [("2025-08-01", 0.64)])

    warm = (lifecycle != "new_visitor") | np.isin(channel, WARM_CHANNELS)
    invited = (dates >= pd.Timestamp("2025-01-10")) & warm & (rng.random(n_rows) < 0.27)
    attendance = (
        0.17
        + 0.14 * owned
        + 0.07 * (lifecycle == "returning_customer")
        + 0.05 * np.isin(segment, ("B2B Services", "Enterprise"))
    )
    attended = invited & (rng.random(n_rows) < np.minimum(attendance, 0.55))

    eligible = np.isin(segment, NEW_PRODUCT_SEGMENTS)
    offer_rate = np.select(
        [dates < pd.Timestamp("2025-10-01"), dates < pd.Timestamp("2026-01-15")], [0.0, 0.18], 0.12
    )
    new_product = eligible & (rng.random(n_rows) < offer_rate)

    quality = (
        52
        + _lookup(channel, CHANNEL_QUALITY)
        + _lookup(segment, SEGMENT_QUALITY)
        + _lookup(lifecycle, LIFECYCLE_QUALITY)
        + _lookup(budget, BUDGET_QUALITY)
        + rng.normal(0, 11, n_rows)
    )

    quarter = dates.year.astype(str) + "Q" + dates.quarter.astype(str)
    return pd.DataFrame(
        {
            "transaction_id": [f"TX-{i:06d}" for i in range(1, n_rows + 1)],
            "date": dates.strftime("%Y-%m-%d"),
            "month": dates.strftime("%Y-%m"),
            "quarter": quarter,
            "channel": channel,
            "campaign_objective": objective,
            "customer_segment": segment,
            "lifecycle_stage": lifecycle,
            "device": device,
            "geo_region": region,
            "ad_budget_level": budget,
            "campaign_daily_spend_eur": np.round(spend, 2),
            "cost_attributed_eur": np.round(cost, 2),
            "landing_variant": np.where(landing_v2 == 1, "landing_v2", "baseline"),
            "cta_variant": np.where(benefit_cta == 1, "benefit_cta", "standard"),
            "lead_magnet": lead_magnet,
            "checkout_simplified": checkout,
            "webinar_invited": invited.astype(int),
            "webinar_attended": attended.astype(int),
            "new_product_offer": new_product.astype(int),
            "lead_score": np.clip(np.round(quality), 1, 99).astype(int),
        }
    )


def true_conversion_probability(frame: pd.DataFrame) -> np.ndarray:
    """P(sale | covariates) under the DGP. Works on any intervened copy of the data."""
    attended = frame["webinar_attended"].to_numpy() == 1
    invited_only = (frame["webinar_invited"].to_numpy() == 1) & ~attended
    logit = (
        INTERCEPT_LOGIT
        + LEAD_SCORE_LOGIT * (frame["lead_score"].to_numpy() - 50)
        + _lookup(frame["channel"], CHANNEL_LOGIT)
        + _lookup(frame["customer_segment"], SEGMENT_LOGIT)
        + _lookup(frame["lifecycle_stage"], LIFECYCLE_LOGIT)
        + _lookup(frame["campaign_objective"], OBJECTIVE_LOGIT)
        + LANDING_V2_LOGIT * (frame["landing_variant"].to_numpy() == "landing_v2")
        + BENEFIT_CTA_LOGIT * (frame["cta_variant"].to_numpy() == "benefit_cta")
        + LEAD_MAGNET_LOGIT * frame["lead_magnet"].to_numpy()
        + CHECKOUT_SIMPLIFIED_LOGIT * frame["checkout_simplified"].to_numpy()
        + WEBINAR_ATTENDED_LOGIT * attended
        + WEBINAR_INVITED_ONLY_LOGIT * invited_only
        + _lookup(frame["ad_budget_level"], BUDGET_LOGIT)
        + NEW_PRODUCT_LOGIT * frame["new_product_offer"].to_numpy()
        + MONTHLY_TREND_LOGIT * month_index(frame["date"])
    )
    return np.clip(1 / (1 + np.exp(-logit)), *CONVERSION_CLIP)


def _median_aov(frame: pd.DataFrame) -> np.ndarray:
    return (
        _lookup(frame["customer_segment"], BASE_AOV)
        * _lookup(frame["channel"], CHANNEL_AOV)
        * np.where(frame["new_product_offer"].to_numpy() == 1, NEW_PRODUCT_AOV_MULTIPLIER, 1.0)
        * np.where(frame["webinar_attended"].to_numpy() == 1, WEBINAR_AOV_MULTIPLIER, 1.0)
    )


def true_expected_aov(frame: pd.DataFrame) -> np.ndarray:
    """E[ticket | sale, covariates] (lognormal mean; clipping is negligible)."""
    return _median_aov(frame) * np.exp(AOV_LOG_SD**2 / 2)


def true_expected_margin(frame: pd.DataFrame) -> np.ndarray:
    """E[gross margin | covariates] (normal mean; clipping is negligible)."""
    return (
        BASE_MARGIN
        + _lookup(frame["channel"], CHANNEL_MARGIN)
        + NEW_PRODUCT_MARGIN_DELTA * frame["new_product_offer"].to_numpy()
    )


def true_expected_value(frame: pd.DataFrame) -> np.ndarray:
    """E[contribution profit | covariates], the quantity the models try to estimate."""
    p = true_conversion_probability(frame)
    value = p * true_expected_aov(frame) * true_expected_margin(frame)
    return value - frame["cost_attributed_eur"].to_numpy() - p * SUPPORT_COST_EUR


def sample_outcomes(covariates: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Draw conversion, ticket, margin and profit for each opportunity."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    n_rows = len(covariates)
    converted = (rng.random(n_rows) < true_conversion_probability(covariates)).astype(int)
    aov = np.round(
        np.clip(rng.lognormal(np.log(_median_aov(covariates)), AOV_LOG_SD), *AOV_CLIP), 2
    )
    margin = np.round(
        np.clip(rng.normal(true_expected_margin(covariates), MARGIN_SD), *MARGIN_CLIP), 3
    )
    attended = covariates["webinar_attended"].to_numpy() == 1
    days = np.maximum(0, rng.gamma(2.0, 3.0, n_rows) - 2 * attended).astype(int)

    revenue = np.round(converted * aov, 2)
    gross_profit = np.round(revenue * margin, 2)
    contribution = np.round(
        gross_profit - covariates["cost_attributed_eur"].to_numpy() - SUPPORT_COST_EUR * converted,
        2,
    )
    outcomes = pd.DataFrame(
        {
            "converted_to_sale": converted,
            "days_to_close": np.where(converted == 1, days, np.nan),
            "aov_eur": np.where(converted == 1, aov, np.nan),
            "revenue_eur": revenue,
            "gross_margin_pct": margin,
            "gross_profit_eur": gross_profit,
            "contribution_profit_eur": contribution,
        },
        index=covariates.index,
    )
    return pd.concat([covariates, outcomes], axis=1)


def generate_dataset(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Full synthetic history: covariates + realised outcomes. Deterministic for a seed."""
    return sample_outcomes(sample_covariates(n_rows, seed), seed)
