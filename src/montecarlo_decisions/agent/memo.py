"""Decision memo: schema, validation and a deterministic rule-based author.

Two authors produce the same memo structure:

* ``claude``: the tool-using agent in :mod:`montecarlo_decisions.agent.claude`;
* ``rules``: :func:`rule_based_memo`, which calls the same tools and fills a fixed
  template with their numbers. It needs no API key and is labelled as such in the UI,
  so a reader always knows whether a language model wrote the text.

Memo text is in Spanish because the dashboard audience is Spanish-speaking.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from montecarlo_decisions.agent.tools import Toolbox, summarize_result
from montecarlo_decisions.scenarios import DECISION_KEYS

LIST_FIELDS = (
    "reasons",
    "watchouts",
    "next_actions",
    "switch_signals",
    "due_diligence",
    "findings",
)
AUDIENCES = ("ceo", "growth", "riesgo")
MAX_ITEMS = 5
TOOL_PURPOSES = {
    "analyze_business": "Lee el historico y resume conversion, ingresos y beneficio por canal.",
    "get_lever_uplift": "Consulta el uplift contrafactual estimado para una palanca.",
    "get_ads_budget_regimes": "Mide la saturacion de paid media por nivel de inversion.",
    "get_decision_distribution": "Lee la distribucion Monte Carlo de una decision.",
    "compare_decisions": "Ordena las alternativas por beneficio esperado y riesgo.",
    "get_validation_checks": "Verifica las comprobaciones de calidad del analisis.",
}

_string_list = {"type": "array", "items": {"type": "string"}}
_audience = {
    "type": "object",
    "properties": {"headline": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["headline", "summary"],
    "additionalProperties": False,
}
MEMO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommended_decision": {"type": "string", "enum": list(DECISION_KEYS)},
        "runner_up": {"type": "string", "enum": list(DECISION_KEYS)},
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        **dict.fromkeys(LIST_FIELDS, _string_list),
        "audience_views": {
            "type": "object",
            "properties": dict.fromkeys(AUDIENCES, _audience),
            "required": list(AUDIENCES),
            "additionalProperties": False,
        },
    },
    "required": [
        "recommended_decision",
        "runner_up",
        "headline",
        "summary",
        *LIST_FIELDS,
        "audience_views",
    ],
    "additionalProperties": False,
}


def validate_memo_content(content: dict[str, Any]) -> dict[str, Any]:
    """Check a memo body against MEMO_SCHEMA plus the rules strict mode cannot express."""
    missing = [key for key in MEMO_SCHEMA["required"] if key not in content]
    if missing:
        raise ValueError(f"Memo is missing fields: {missing}")
    for key in ("recommended_decision", "runner_up"):
        if content[key] not in DECISION_KEYS:
            raise ValueError(f"{key}={content[key]!r} is not a known decision")
    if content["recommended_decision"] == content["runner_up"]:
        raise ValueError("runner_up must differ from recommended_decision")
    for key in ("headline", "summary"):
        if not isinstance(content[key], str) or not content[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    clean: dict[str, Any] = {
        "recommended_decision": content["recommended_decision"],
        "runner_up": content["runner_up"],
        "headline": content["headline"].strip(),
        "summary": content["summary"].strip(),
    }
    for key in LIST_FIELDS:
        items = [str(item).strip() for item in content[key] if str(item).strip()]
        if not items:
            raise ValueError(f"{key} must contain at least one item")
        clean[key] = items[:MAX_ITEMS]
    views = content["audience_views"]
    clean["audience_views"] = {}
    for audience in AUDIENCES:
        view = views.get(audience) or {}
        if not str(view.get("headline", "")).strip() or not str(view.get("summary", "")).strip():
            raise ValueError(f"audience_views.{audience} needs headline and summary")
        clean["audience_views"][audience] = {
            "headline": str(view["headline"]).strip(),
            "summary": str(view["summary"]).strip(),
        }
    return clean


def trace_entry(name: str, arguments: dict[str, Any], outcome: str) -> dict[str, str]:
    args = ", ".join(f"{key}={value}" for key, value in arguments.items()) or "sin parametros"
    return {
        "name": name,
        "purpose": TOOL_PURPOSES.get(name, "Consulta del agente."),
        "args": args,
        "outcome": outcome,
    }


def finalize_memo(
    content: dict[str, Any],
    *,
    source: str,
    model: str | None,
    fingerprint: str,
    tool_trace: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "source": source,
        "model": model,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        **validate_memo_content(content),
        "tool_trace": tool_trace,
    }


def _eur(value: float) -> str:
    return f"{value:,.0f} EUR".replace(",", ".")


def rule_based_memo(toolbox: Toolbox) -> dict[str, Any]:
    """Deterministic memo built only from tool results (no language model)."""
    trace: list[dict[str, str]] = []

    def call(name: str, **arguments: Any) -> dict[str, Any]:
        result = toolbox.call(name, arguments)
        trace.append(trace_entry(name, arguments, summarize_result(name, result)))
        return result

    ranking = call("compare_decisions")["ranking"]
    distributions = {
        row["decision"]: call("get_decision_distribution", decision=row["decision"])
        for row in ranking
    }
    funnel = call("get_lever_uplift", lever="funnel")
    regimes = {row["ad_budget_level"]: row for row in call("get_ads_budget_regimes")["regimes"]}
    checks = call("get_validation_checks")["checks"]

    best, second = ranking[0], ranking[1]
    best_dist, second_dist = distributions[best["decision"]], distributions[second["decision"]]
    riskiest = max(ranking, key=lambda row: row["probability_loss"])
    widest = max(ranking, key=lambda row: row["p90_eur"] - row["p10_eur"])
    losers = [row for row in ranking if row["expected_profit_eur"] < 0]
    gap = best["expected_profit_eur"] - second["expected_profit_eur"]
    checks_passed = sum(check["passed"] for check in checks)
    saturation = ", ".join(
        f"{level} {row['contribution_per_opportunity_eur']:.1f}" for level, row in regimes.items()
    )
    low, high = funnel["value_lift_95ci_eur"]

    watchouts = [
        f"{riskiest['label']} es la opcion con mas riesgo: {riskiest['probability_loss']:.1%} de "
        f"probabilidad de perder dinero y CVaR 5% de {_eur(riskiest['cvar_5_eur'])}.",
        f"{widest['label']} tiene la mayor dispersion (P10 {_eur(widest['p10_eur'])}, "
        f"P90 {_eur(widest['p90_eur'])}): su resultado depende mas de la ejecucion que del modelo.",
    ]
    if losers:
        names = ", ".join(row["label"] for row in losers)
        watchouts.append(
            f"Beneficio esperado negativo en: {names}. No deberian financiarse tal cual."
        )
    else:
        watchouts.append(
            "Ninguna alternativa tiene beneficio esperado negativo con los supuestos actuales."
        )

    content = {
        "recommended_decision": best["decision"],
        "runner_up": second["decision"],
        "headline": f"Priorizar {best['label']}: mayor beneficio esperado con el riesgo acotado.",
        "summary": (
            f"{best['label']} genera {_eur(best['expected_profit_eur'])} esperados en el "
            f"horizonte, con P(perdida) de {best['probability_loss']:.1%} y es la mejor opcion "
            f"en el {best['probability_best']:.0%} de los futuros simulados. {second['label']} "
            f"queda segunda a {_eur(gap)} de distancia."
        ),
        "reasons": [
            f"Mayor beneficio esperado ({_eur(best['expected_profit_eur'])}) y P10 de "
            f"{_eur(best['p10_eur'])}.",
            f"Seguiria siendo rentable salvo que la probabilidad de fracaso supere el "
            f"{best_dist['breakeven_failure_probability']:.0%} (supuesto actual: "
            f"{best_dist['assumptions']['failure_probability']:.0%}).",
            "El uplift del funnel estimado por el modelo es de "
            f"{funnel['value_lift_per_opportunity_eur']:.1f} EUR por oportunidad "
            f"(IC95 {low:.1f}-{high:.1f}), con control del efecto calendario.",
        ],
        "watchouts": watchouts,
        "next_actions": [
            f"Lanzar {best['label']} por fases, con metrica de conversion por etapa y un owner.",
            f"Preparar {second['label']} como segunda apuesta si la primera no confirma el uplift.",
            "Revisar el resultado real frente a la simulacion tras el primer ciclo y "
            "actualizar los supuestos.",
        ],
        "switch_signals": [
            f"Rotar a {second['label']} si el uplift observado cae por debajo del limite "
            "inferior del IC95.",
            "Reconsiderar el ranking si la probabilidad de fracaso real de "
            f"{best['label']} se acerca al {best_dist['breakeven_failure_probability']:.0%}.",
            "No escalar paid media mientras la contribucion por oportunidad siga cayendo con "
            f"el nivel de inversion (EUR/op por nivel: {saturation}).",
        ],
        "due_diligence": [
            "Confirmar que los costes fijos y probabilidades de fracaso de scenarios.toml "
            "reflejan el plan real.",
            f"Validar con negocio el supuesto de ejecucion de {second_dist['label']} "
            f"(fracaso {second_dist['assumptions']['failure_probability']:.0%}).",
            "Asegurar medicion por experimento (A/B) para separar el efecto de la palanca "
            "de la tendencia.",
        ],
        "findings": [
            f"{checks_passed}/{len(checks)} comprobaciones de calidad del analisis superadas.",
            f"{best['label']} devuelve {best['expected_roi']:.1f} EUR de beneficio incremental por "
            "cada euro de coste fijo.",
            "La contribucion por oportunidad de paid media cae al subir el nivel de inversion: "
            f"{saturation} EUR.",
        ],
        "audience_views": {
            "ceo": {
                "headline": f"Asignar el presupuesto a {best['label']}.",
                "summary": "Es la opcion con mejor retorno esperado "
                f"({_eur(best['expected_profit_eur'])}) y probabilidad de perdida del "
                f"{best['probability_loss']:.1%}.",
            },
            "growth": {
                "headline": f"{best['label']} primero, {second['label']} despues.",
                "summary": "Secuenciar las palancas permite medir cada uplift por separado "
                "y escalar solo lo que se confirma.",
            },
            "riesgo": {
                "headline": f"Evitar {riskiest['label']} sin controles adicionales.",
                "summary": "Concentra el mayor riesgo de perdida "
                f"({riskiest['probability_loss']:.1%}); la recomendada tiene un CVaR 5% de "
                f"{_eur(best['cvar_5_eur'])}.",
            },
        },
    }
    return finalize_memo(
        content,
        source="rules",
        model=None,
        fingerprint=toolbox.artifacts.fingerprint,
        tool_trace=trace,
    )


def load_memo(path: Path, fingerprint: str) -> dict[str, Any] | None:
    """Return a saved memo only if it was written for the current simulation results."""
    try:
        memo = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return memo if memo.get("fingerprint") == fingerprint else None
