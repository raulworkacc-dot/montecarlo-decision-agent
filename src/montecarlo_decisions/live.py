"""Live status of a running simulation, shared with the dashboards.

The status is written as a small JavaScript file (``window.__MONTECARLO_STATUS__ = ...``)
so the live dashboard can poll it with a ``<script>`` tag. That works both through the
local server and when the HTML is opened straight from disk (``file://``), where
``fetch`` of a JSON file is blocked by browsers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from montecarlo_decisions.simulation import summarize

STATUS_PREFIX = "window.__MONTECARLO_STATUS__ = "
RECENT_RUNS = 40  # ~10 recent futures per decision for the sparklines
REPLACE_RETRIES = 25


def build_status(
    current: int,
    total: int,
    simulations: pd.DataFrame,
    labels: dict[str, str],
    phase: str = "running",
    message: str = "",
) -> dict[str, object]:
    leaderboard: list[dict[str, object]] = []
    recent: list[dict[str, object]] = []
    if not simulations.empty:
        summary = summarize(simulations, labels)
        leaderboard = [
            {
                "ranking": int(row.ranking),
                "key": row.decision,
                "decision": row.label,
                "expected_profit_eur": round(float(row.expected_profit_eur), 2),
                "p10_eur": round(float(row.p10_eur), 2),
                "p90_eur": round(float(row.p90_eur), 2),
                "probability_loss": round(float(row.probability_loss), 4),
                "expected_roi": round(float(row.expected_roi), 4),
            }
            for row in summary.itertuples()
        ]
        tail = simulations.sort_values("simulation", kind="stable").tail(RECENT_RUNS)
        recent = [
            {
                "simulation": int(row.simulation),
                "decision": labels.get(row.decision, row.decision),
                "incremental_profit_eur": round(float(row.incremental_profit_eur), 2),
                "roi": round(float(row.incremental_profit_eur / row.cost_eur), 4),
            }
            for row in tail.itertuples()
        ]
    if phase == "running" and current >= total:
        phase = "finalizing"  # all futures drawn; results are being written
    return {
        "current_simulation": int(current),
        "total_simulations": int(total),
        "progress_pct": round(100 * current / max(1, total), 2),
        "is_running": phase == "running" and current < total,
        "phase": phase,
        "status_message": message,
        "leaderboard": leaderboard,
        "recent_runs": recent,
    }


def idle_status(total: int, message: str = "Esperando lanzamiento.") -> dict[str, object]:
    return build_status(0, total, pd.DataFrame(), {}, phase="idle", message=message)


def write_status(path: Path, status: dict[str, object]) -> None:
    """Atomic write: readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(STATUS_PREFIX + json.dumps(status, ensure_ascii=False) + ";", encoding="utf-8")
    for attempt in range(REPLACE_RETRIES):
        try:
            tmp.replace(path)
            return
        except PermissionError:  # Windows: a reader holds the file open for a moment
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(0.02)


def read_status(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text.startswith(STATUS_PREFIX):
        return None
    return json.loads(text.removeprefix(STATUS_PREFIX).rstrip(";"))
