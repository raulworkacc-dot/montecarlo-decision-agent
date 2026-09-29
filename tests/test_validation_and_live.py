import numpy as np
import pandas as pd

from montecarlo_decisions import live, validation


def test_pipeline_checks_pass(result):
    failed = [check for check in result.checks if not check.passed]
    assert not failed, failed
    assert [check.name for check in result.checks] == [
        "data_integrity",
        "conversion_model_discrimination",
        "conversion_model_calibration",
        "counterfactual_recovery",
        "ranking_resolved",
    ]


def test_data_integrity_detects_problems(prepared):
    broken = prepared.history.head(100).copy()
    broken.loc[broken.index[0], "cost_attributed_eur"] = -1
    broken.loc[broken.index[1], "transaction_id"] = broken["transaction_id"].iloc[2]
    check = validation.check_data_integrity(broken)
    assert not check.passed
    assert "negative acquisition cost" in check.detail
    assert "duplicated transaction_id" in check.detail


def test_ranking_is_unresolved_for_identical_decisions():
    rng = np.random.default_rng(0)
    profit = rng.normal(0, 100, 500)
    sims = pd.DataFrame(
        {
            "simulation": np.tile(np.arange(500), 2),
            "decision": np.repeat(["a", "b"], 500),
            "incremental_profit_eur": np.concatenate([profit, profit + rng.normal(0, 1, 500)]),
        }
    )
    summary = pd.DataFrame({"decision": ["a", "b"]})
    assert not validation.check_ranking_resolved(sims, summary).passed


def test_status_round_trip(tmp_path, result):
    labels = result.prepared.config.labels()
    status = live.build_status(500, 2_000, result.simulations.head(2_000), labels)
    assert status["is_running"]
    assert status["progress_pct"] == 25.0
    assert len(status["recent_runs"]) == live.RECENT_RUNS
    assert {row["decision"] for row in status["leaderboard"]} <= set(labels.values())
    path = tmp_path / "status.js"
    live.write_status(path, status)
    assert path.read_text(encoding="utf-8").startswith(live.STATUS_PREFIX)
    assert live.read_status(path) == status


def test_idle_status_and_missing_file(tmp_path):
    status = live.idle_status(100)
    assert status["phase"] == "idle"
    assert not status["is_running"]
    assert status["leaderboard"] == []
    assert live.read_status(tmp_path / "missing.js") is None


def test_completed_status_is_not_running(result):
    labels = result.prepared.config.labels()
    status = live.build_status(10, 10, result.simulations, labels, phase="completed")
    assert not status["is_running"]


def test_recent_runs_cover_every_decision(result):
    labels = result.prepared.config.labels()
    status = live.build_status(2_000, 2_000, result.simulations, labels)
    assert {row["decision"] for row in status["recent_runs"]} == set(labels.values())
    assert status["phase"] == "finalizing"
    assert not status["is_running"]
