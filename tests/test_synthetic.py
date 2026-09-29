import numpy as np
import pandas as pd
import pytest

from montecarlo_decisions import synthetic
from montecarlo_decisions.scenarios import FUNNEL_TREATMENT


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return synthetic.generate_dataset()


def test_generation_is_deterministic_for_a_seed():
    first = synthetic.generate_dataset(n_rows=500, seed=7)
    second = synthetic.generate_dataset(n_rows=500, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_histories():
    first = synthetic.generate_dataset(n_rows=500, seed=1)
    second = synthetic.generate_dataset(n_rows=500, seed=2)
    assert not first["converted_to_sale"].equals(second["converted_to_sale"])


def test_shape_and_period(history):
    assert len(history) == synthetic.N_ROWS
    assert history["transaction_id"].is_unique
    assert history["date"].min() >= synthetic.START_DATE
    assert history["date"].max() <= synthetic.END_DATE
    assert history["date"].is_monotonic_increasing


def test_ticket_is_present_exactly_for_sales(history):
    sold = history["converted_to_sale"] == 1
    assert history.loc[sold, "aov_eur"].notna().all()
    assert history.loc[~sold, "aov_eur"].isna().all()
    assert (history.loc[~sold, "revenue_eur"] == 0).all()


def test_rollout_schedule_is_respected(history):
    assert (history.loc[history["date"] < "2024-09-15", "landing_variant"] == "baseline").all()
    assert (history.loc[history["date"] < "2025-03-15", "lead_magnet"] == 0).all()
    assert (history.loc[history["date"] < "2025-08-01", "checkout_simplified"] == 0).all()
    assert (history.loc[history["date"] < "2025-10-01", "new_product_offer"] == 0).all()
    assert (history["webinar_attended"] <= history["webinar_invited"]).all()


def test_budget_levels_only_apply_to_paid_channels(history):
    paid = history["channel"].isin(synthetic.PAID_CHANNELS)
    assert (history.loc[~paid, "ad_budget_level"] == "organic_or_owned").all()
    assert (history.loc[~paid, "campaign_daily_spend_eur"] == 0).all()
    assert (history.loc[paid, "ad_budget_level"] != "organic_or_owned").all()


def test_new_product_only_for_eligible_segments(history):
    offered = history[history["new_product_offer"] == 1]
    assert offered["customer_segment"].isin(synthetic.NEW_PRODUCT_SEGMENTS).all()


def test_true_conversion_matches_observed_rate(history):
    expected = synthetic.true_conversion_probability(history).mean()
    assert expected == pytest.approx(history["converted_to_sale"].mean(), abs=0.01)


def test_true_expected_value_matches_observed_profit(history):
    expected = synthetic.true_expected_value(history).mean()
    observed = history["contribution_profit_eur"].mean()
    assert expected == pytest.approx(observed, rel=0.05)


def test_funnel_treatment_raises_true_conversion_everywhere(history):
    treated = history.assign(**FUNNEL_TREATMENT)
    before = synthetic.true_conversion_probability(history)
    after = synthetic.true_conversion_probability(treated)
    assert np.all(after >= before)
    assert after.mean() > before.mean()


def test_calendar_trend_is_part_of_the_dgp(history):
    later = history.assign(date="2026-04-30")
    earlier = history.assign(date="2024-01-01")
    assert (
        synthetic.true_conversion_probability(later).mean()
        > synthetic.true_conversion_probability(earlier).mean()
    )


def test_unknown_category_raises():
    frame = synthetic.sample_covariates(n_rows=5).assign(channel="TikTok")
    with pytest.raises(KeyError, match="TikTok"):
        synthetic.true_conversion_probability(frame)


def test_month_index_starts_at_zero():
    dates = pd.Series(["2024-01-01", "2024-12-31", "2026-04-30"])
    assert synthetic.month_index(dates).tolist() == [0, 11, 27]
