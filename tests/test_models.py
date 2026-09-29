import numpy as np
import pytest

from montecarlo_decisions import models
from montecarlo_decisions.synthetic import generate_dataset


def test_conversion_features_exclude_costs_and_include_time():
    assert "month_index" in models.FEATURES
    assert "cost_attributed_eur" not in models.FEATURES
    assert "campaign_daily_spend_eur" not in models.FEATURES
    assert "month_index" not in models.NAIVE_FEATURES


def test_with_time_index_is_idempotent():
    frame = models.with_time_index(generate_dataset(n_rows=50))
    assert models.with_time_index(frame) is frame
    assert frame["month_index"].between(0, 27).all()


def test_temporal_split_trains_on_the_past():
    frame = generate_dataset(n_rows=1_000)
    train, test = models.temporal_split(frame, holdout_fraction=0.2)
    assert len(train) == 800
    assert len(test) == 200
    assert train["date"].max() <= test["date"].min()


def test_gain_chart_is_cumulative_and_complete():
    rng = np.random.default_rng(0)
    actual = rng.integers(0, 2, 1_000)
    chart = models.gain_chart(actual, rng.random(1_000))
    captures = [row["capture_pct"] for row in chart]
    assert captures == sorted(captures)
    assert captures[-1] == pytest.approx(1.0)
    assert chart[-1]["population_pct"] == pytest.approx(1.0)


def test_perfect_ranking_captures_conversions_first():
    actual = np.array([1] * 100 + [0] * 900)
    chart = models.gain_chart(actual, actual.astype(float))
    assert chart[0]["capture_pct"] == pytest.approx(1.0)


def test_calibration_error_is_zero_for_calibrated_scores():
    score = np.repeat(np.linspace(0.05, 0.95, 10), 100)
    actual = np.concatenate(
        [np.r_[np.ones(round(p * 100)), np.zeros(100 - round(p * 100))] for p in score[::100]]
    )
    table = models.calibration_table(actual, score)
    assert models.expected_calibration_error(table) == pytest.approx(0.0, abs=1e-9)


def test_ticket_model_uses_smearing(prepared):
    ticket = prepared.model.ticket
    assert ticket.smearing > 1.0
    predictions = ticket.predict(prepared.history.head(200))
    assert (predictions > 0).all()


def test_expected_value_decomposition(prepared):
    frame = prepared.history.head(300)
    parts = prepared.model.components(frame)
    assert ((parts.conversion > 0) & (parts.conversion < 1)).all()
    assert ((parts.margin > 0.4) & (parts.margin < 1)).all()
    np.testing.assert_allclose(
        parts.expected_value, prepared.model.expected_value(frame), rtol=1e-12
    )


def test_out_of_time_metrics(prepared):
    report = prepared.model_report
    assert report["split"]["strategy"] == "temporal"
    assert report["classification"]["auc"] > 0.68
    assert report["classification"]["expected_calibration_error"] < 0.03
    assert report["regression"]["r2"] > 0.5


def test_bootstrap_rows_are_reproducible():
    first = models.bootstrap_rows(100, 3, seed=1)
    second = models.bootstrap_rows(100, 3, seed=1)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
