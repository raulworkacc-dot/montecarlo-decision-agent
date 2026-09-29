"""The core methodological claims of the case, tested against the synthetic truth."""

import numpy as np
import pandas as pd
import pytest

from montecarlo_decisions.counterfactuals import add_bootstrap_intervals


@pytest.fixture(scope="module")
def uplift(prepared) -> pd.DataFrame:
    return prepared.uplift.set_index("lever")


def test_every_lever_is_reported(uplift):
    assert list(uplift.index) == ["funnel", "webinar", "new_product"]


def test_truth_lies_inside_the_bootstrap_interval(uplift):
    assert uplift["truth_in_interval"].all(), uplift[
        ["value_lift_ci_low_eur", "value_lift_ci_high_eur", "true_value_lift_per_opportunity_eur"]
    ]


def test_time_control_removes_the_funnel_bias(uplift):
    funnel = uplift.loc["funnel"]
    assert abs(funnel["value_lift_error_pct"]) < 0.10
    assert funnel["naive_value_lift_error_pct"] > 0.10  # calendar drift credited to the funnel
    assert abs(funnel["naive_value_lift_error_pct"]) > abs(funnel["value_lift_error_pct"])


def test_signs_match_the_dgp(uplift):
    assert uplift.loc["funnel", "value_lift_per_opportunity_eur"] > 0
    assert uplift.loc["webinar", "conversion_lift_pct"] > 0
    assert uplift.loc["new_product", "true_conversion_lift_pct"] < 0  # higher ticket, fewer sales


def test_bootstrap_interval_is_percentile_based():
    effects = pd.DataFrame({"true_value_lift_per_opportunity_eur": [5.0, 50.0]})
    lifts = np.column_stack([np.arange(1, 101), np.arange(1, 101)]).astype(float)
    result = add_bootstrap_intervals(effects, lifts, level=0.90)
    assert result["value_lift_ci_low_eur"].tolist() == pytest.approx([5.95, 5.95])
    assert result["value_lift_ci_high_eur"].tolist() == pytest.approx([95.05, 95.05])
    assert result["truth_in_interval"].tolist() == [False, True]
