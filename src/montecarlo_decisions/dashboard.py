"""Mission Control dashboard: payload assembly and HTML rendering.

The same HTML works in two modes:

* **served** by :mod:`montecarlo_decisions.server`: stage buttons call the local API and
  the Monte Carlo stage launches a real simulation;
* **static** (``mcd dashboard``): the payload is embedded and the page runs from disk or
  from any static host (e.g. GitHub Pages) without a backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from montecarlo_decisions.agent.memo import load_memo, rule_based_memo
from montecarlo_decisions.agent.tools import Toolbox
from montecarlo_decisions.artifacts import Artifacts
from montecarlo_decisions.config import TEMPLATES_DIR
from montecarlo_decisions.live import build_status, write_status

DECISION_COLORS = {
    "funnel": "#27f5c6",
    "webinar": "#58a6ff",
    "ads": "#ffd166",
    "new_product": "#ff5c8a",
}
LINE_SEPARATOR, PARAGRAPH_SEPARATOR = chr(0x2028), chr(0x2029)  # invalid raw in old JS
SAMPLE_COLUMNS = [
    "date",
    "channel",
    "customer_segment",
    "campaign_objective",
    "lead_score",
    "converted_to_sale",
    "revenue_eur",
    "contribution_profit_eur",
]
STAGES: list[dict[str, Any]] = [
    {
        "key": "briefing",
        "title": "Introduccion",
        "eyebrow": "Fase 01",
        "tagline": "Objetivo del caso, metodo y comprobaciones de calidad del analisis.",
        "command": "AGENTE.INICIAR_PRESENTACION()",
        "duration_ms": 1100,
    },
    {
        "key": "ingesta",
        "title": "Carga de datos",
        "eyebrow": "Fase 02",
        "tagline": "Historico de oportunidades: cobertura, estructura y evolucion del negocio.",
        "command": "AGENTE.CARGAR_Y_VALIDAR_DATOS()",
        "duration_ms": 1400,
    },
    {
        "key": "uplift",
        "title": "Modelos y uplift",
        "eyebrow": "Fase 03",
        "tagline": "Calidad de los modelos y uplift contrafactual de cada palanca, validado.",
        "command": "AGENTE.EVALUAR_MODELOS_Y_PALANCAS()",
        "duration_ms": 1500,
    },
    {
        "key": "montecarlo",
        "title": "Monte Carlo",
        "eyebrow": "Fase 04",
        "tagline": "Simulacion en directo de miles de futuros y ranking por retorno y riesgo.",
        "command": "AGENTE.SIMULAR_FUTUROS()",
        "duration_ms": 1600,
    },
    {
        "key": "reporte",
        "title": "Informe final",
        "eyebrow": "Fase 05",
        "tagline": "Recomendacion del agente, riesgos, senales de cambio y due diligence.",
        "command": "AGENTE.EMITIR_RECOMENDACION()",
        "duration_ms": 1500,
    },
]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe records (NaN -> None, numpy scalars -> Python)."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def current_memo(artifacts: Artifacts) -> dict[str, Any]:
    """The saved agent memo if it matches these results, else a fresh rule-based one."""
    saved = load_memo(artifacts.paths.agent_memo, artifacts.fingerprint)
    return saved or rule_based_memo(Toolbox(artifacts))


def build_payload(
    artifacts: Artifacts,
    memo: dict[str, Any] | None = None,
    pace_seconds: float = 0.0,
    mode: str = "static",
) -> dict[str, Any]:
    """Everything the dashboard renders. ``mode`` is "server" only for the local app."""
    memo = memo or current_memo(artifacts)
    dataset = artifacts.dataset
    converted = dataset[dataset["converted_to_sale"] == 1]
    summary = artifacts.summary.sort_values("ranking")
    labels = artifacts.config.labels()
    best = summary.iloc[0]
    recommended = summary.set_index("decision").loc[memo["recommended_decision"]]
    runner_up = summary.set_index("decision").loc[memo["runner_up"]]

    monthly = (
        dataset.groupby("month")
        .agg(
            revenue_eur=("revenue_eur", "sum"),
            contribution_profit_eur=("contribution_profit_eur", "sum"),
            conversion_rate=("converted_to_sale", "mean"),
        )
        .reset_index()
    )
    return {
        "mode": mode,
        "title": "Agente de Decision Comercial",
        "subtitle": (
            "Analiza el historico, estima el uplift de cada iniciativa, simula miles de futuros "
            "y emite una recomendacion trazable antes de decidir."
        ),
        "stages": STAGES,
        "dataset": {
            "summary": {
                "rows": len(dataset),
                "period_start": str(dataset["date"].min()),
                "period_end": str(dataset["date"].max()),
                "conversion_rate": float(dataset["converted_to_sale"].mean()),
                "revenue_eur": float(dataset["revenue_eur"].sum()),
                "contribution_profit_eur": float(dataset["contribution_profit_eur"].sum()),
                "avg_ticket_eur": float(converted["aov_eur"].mean()),
                "avg_lead_score": float(dataset["lead_score"].mean()),
                "campaign_count": int(
                    dataset[["channel", "campaign_objective"]].drop_duplicates().shape[0]
                ),
                "variable_count": len(dataset.columns),
                "channel_count": int(dataset["channel"].nunique()),
                "segment_count": int(dataset["customer_segment"].nunique()),
                "geography_count": int(dataset["geo_region"].nunique()),
                "time_windows": int(dataset["month"].nunique()),
            },
            "monthly": _records(monthly),
            "sample": _records(dataset[SAMPLE_COLUMNS].head(10)),
        },
        "models": artifacts.model_report,
        "uplift": {
            "main": _records(artifacts.uplift),
            "ads": _records(artifacts.ads_regimes),
        },
        "simulation": {
            "summary": _records(summary),
            "total_simulations": int(artifacts.simulations["simulation"].max()),
            "live_dashboard": "live.html",
            "pace_seconds": pace_seconds,
            "labels": labels,
        },
        "recommendation": {
            "decision": memo["recommended_decision"],
            "label": labels[memo["recommended_decision"]],
            "runner_up": labels[memo["runner_up"]],
            "headline": memo["headline"],
            "agent_summary": memo["summary"],
            "expected_profit_eur": float(recommended["expected_profit_eur"]),
            "expected_roi": float(recommended["expected_roi"]),
            "probability_loss": float(recommended["probability_loss"]),
            "gap_vs_runner_up_eur": float(
                recommended["expected_profit_eur"] - runner_up["expected_profit_eur"]
            ),
            "matches_simulation_leader": memo["recommended_decision"] == best["decision"],
            "reasons": memo["reasons"],
            "watchouts": memo["watchouts"],
            "next_actions": memo["next_actions"],
            "switch_signals": memo["switch_signals"],
            "due_diligence": memo["due_diligence"],
            "findings": memo["findings"],
            "tool_trace": memo["tool_trace"],
            "audience_views": memo["audience_views"],
            "memo_source": {
                "author": memo["source"],
                "model": memo.get("model"),
                "generated_at": memo.get("generated_at"),
            },
        },
        "report": {"checks": artifacts.checks},
    }


def _template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _embed_json(value: Any) -> str:
    """JSON safe to inline inside a <script> element."""
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        .replace("</", "<\\/")
        .replace(LINE_SEPARATOR, "\\u2028")
        .replace(PARAGRAPH_SEPARATOR, "\\u2029")
    )


def render_mission_control(payload: dict[str, Any]) -> str:
    return _template("mission_control.html").replace("__PAYLOAD__", _embed_json(payload))


def render_live_dashboard(labels: dict[str, str]) -> str:
    colors = {labels[key]: color for key, color in DECISION_COLORS.items() if key in labels}
    return _template("live.html").replace("__COLORS__", _embed_json(colors))


def completed_status(artifacts: Artifacts) -> dict[str, Any]:
    total = int(artifacts.simulations["simulation"].max())
    return build_status(
        total,
        total,
        artifacts.simulations,
        artifacts.config.labels(),
        phase="completed",
        message="Simulacion completada.",
    )


def build_static_site(artifacts: Artifacts, output_dir: Path) -> Path:
    """Write a self-contained dashboard (index.html, live.html, live_status.js)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index = output_dir / "index.html"
    index.write_text(render_mission_control(build_payload(artifacts)), encoding="utf-8")
    (output_dir / "live.html").write_text(
        render_live_dashboard(artifacts.config.labels()), encoding="utf-8"
    )
    write_status(output_dir / "live_status.js", completed_status(artifacts))
    return index
