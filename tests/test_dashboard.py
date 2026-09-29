import json
import re

import pytest

from montecarlo_decisions import dashboard
from montecarlo_decisions.agent.memo import rule_based_memo
from montecarlo_decisions.agent.tools import Toolbox


@pytest.fixture(scope="module")
def payload(artifacts) -> dict:
    return dashboard.build_payload(artifacts)


def test_payload_is_strict_json(payload):
    json.dumps(payload, allow_nan=False)


def test_payload_contract_used_by_the_frontend(payload):
    assert {
        "title",
        "stages",
        "dataset",
        "models",
        "uplift",
        "simulation",
        "recommendation",
    } <= set(payload)
    assert [stage["key"] for stage in payload["stages"]] == [
        "briefing",
        "ingesta",
        "uplift",
        "montecarlo",
        "reporte",
    ]
    assert payload["models"]["classification"]["gain_chart"]
    assert payload["recommendation"]["memo_source"]["author"] == "rules"
    assert payload["simulation"]["live_dashboard"] == "live.html"


def test_recommendation_uses_labels(payload, artifacts):
    labels = artifacts.config.labels()
    recommendation = payload["recommendation"]
    assert recommendation["label"] == labels[recommendation["decision"]]
    assert recommendation["matches_simulation_leader"]


def test_embedded_payload_cannot_break_out_of_the_script(artifacts):
    memo = rule_based_memo(Toolbox(artifacts))
    memo["headline"] = "</script><script>alert(1)</script>"
    html = dashboard.render_mission_control(dashboard.build_payload(artifacts, memo))
    assert "__PAYLOAD__" not in html
    assert "</script><script>alert(1)" not in html
    assert "<\\/script><script>alert(1)<\\/script>" in html


def test_templates_escape_dynamic_text():
    template = (dashboard.TEMPLATES_DIR / "mission_control.html").read_text(encoding="utf-8")
    assert "function esc(" in template
    for field in ("item.outcome", "item.name", "row.label", "row.description"):
        assert re.search(rf"\$\{{esc\({re.escape(field)}\)\}}", template), field
    assert "tuprimerasemana" not in template


def test_live_dashboard_gets_decision_colors(artifacts):
    html = dashboard.render_live_dashboard(artifacts.config.labels())
    assert "__COLORS__" not in html
    assert "Webinar de ventas" in html
    assert "/api/debug-log" not in html


def test_static_site(tmp_path, artifacts):
    index = dashboard.build_static_site(artifacts, tmp_path / "site")
    names = sorted(path.name for path in index.parent.iterdir())
    assert names == ["index.html", "live.html", "live_status.js"]
    status = (tmp_path / "site" / "live_status.js").read_text(encoding="utf-8")
    assert '"phase": "completed"' in status
