"""HTTP API and security surface of the local web app."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from montecarlo_decisions import server as server_module
from montecarlo_decisions.agent.claude import AgentError
from montecarlo_decisions.agent.memo import rule_based_memo
from montecarlo_decisions.agent.tools import Toolbox
from montecarlo_decisions.server import MAX_BODY_BYTES, MissionControl, make_server


@pytest.fixture(scope="module")
def server(prepared, artifacts):
    app = MissionControl(
        paths=artifacts.paths,
        n_simulations=500,
        pace_seconds=0.0,
        use_claude=False,
        config=prepared.config,
        prepared=prepared,
    )
    app.bootstrap()
    httpd = make_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", app
    httpd.shutdown()
    httpd.server_close()


def request(url: str, body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def post_stage(base: str, payload: object):
    return request(
        f"{base}/api/stage",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )


def test_index_embeds_the_payload(server):
    base, _ = server
    status, headers, body = request(f"{base}/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert b"__PAYLOAD__" not in body


def test_live_assets(server):
    base, _ = server
    assert request(f"{base}/live.html")[0] == 200
    status, headers, body = request(f"{base}/live_status.js")
    assert status == 200
    assert headers["Content-Type"].startswith("text/javascript")
    assert body.startswith(b"window.__MONTECARLO_STATUS__")


@pytest.mark.parametrize(
    "path", ["/.env", "/pyproject.toml", "/../pyproject.toml", "/data/transactions.csv", "/%2e%2e/"]
)
def test_only_allow_listed_routes_are_served(server, path):
    base, _ = server
    assert request(f"{base}{path}")[0] == 404


def test_stage_validation(server):
    base, _ = server
    assert post_stage(base, {"stage": "rm -rf"})[0] == 400
    assert post_stage(base, ["briefing"])[0] == 400
    bad_json = request(f"{base}/api/stage", b"{not json", {"Content-Type": "application/json"})
    assert bad_json[0] == 400
    json_header = {"Content-Type": "application/json"}
    too_big = request(f"{base}/api/stage", b"x" * (MAX_BODY_BYTES + 1), json_header)
    assert too_big[0] == 400
    assert post_stage(base, {"stage": ["x"]})[0] == 400  # unhashable stage
    assert request(f"{base}/api/stage", b"\xff\xfe", json_header)[0] == 400  # invalid UTF-8
    assert request(f"{base}/api/unknown", b"{}")[0] == 404


def test_cross_origin_and_non_json_posts_are_rejected(server):
    base, _ = server
    body = json.dumps({"stage": "briefing"}).encode()
    evil = {"Content-Type": "application/json", "Origin": "https://evil.example"}
    assert request(f"{base}/api/stage", body, evil)[0] == 403
    form = {"Content-Type": "text/plain"}
    assert request(f"{base}/api/stage", body, form)[0] == 415
    same = {"Content-Type": "application/json", "Origin": base}
    assert request(f"{base}/api/stage", body, same)[0] == 200


def test_report_stage_returns_the_rule_based_memo(server):
    base, _ = server
    status, _, body = post_stage(base, {"stage": "reporte"})
    data = json.loads(body)
    assert status == 200
    assert data["payload"]["recommendation"]["memo_source"]["author"] == "rules"
    assert any("reglas" in line for line in data["logs"])


def test_montecarlo_stage_runs_a_live_simulation(server):
    base, app = server
    status, _, body = post_stage(base, {"stage": "montecarlo"})
    assert status == 200
    assert json.loads(body)["status"] == "running"

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = json.loads(request(f"{base}/api/montecarlo-status")[2])
        if state["phase"] == "completed":
            break
        time.sleep(0.2)
    assert state["phase"] == "completed"
    assert state["payload"]["simulation"]["total_simulations"] == 500
    assert state["progress_pct"] == 100.0
    assert app.error is None


def test_claude_memo_is_generated_once_and_cached(prepared, artifacts, monkeypatch):
    calls = []

    def fake_write_memo(toolbox):
        calls.append(1)
        memo = rule_based_memo(toolbox)
        return {**memo, "source": "claude", "model": "claude-opus-5"}

    monkeypatch.setattr(server_module, "write_memo", fake_write_memo)
    app = MissionControl(
        paths=artifacts.paths,
        n_simulations=500,
        pace_seconds=0.0,
        use_claude=True,
        config=prepared.config,
        prepared=prepared,
    )
    try:
        first = app.stage("reporte")
        second = app.stage("reporte")
    finally:
        artifacts.paths.agent_memo.unlink(missing_ok=True)
    assert first["payload"]["recommendation"]["memo_source"]["author"] == "claude"
    assert second["payload"]["recommendation"]["memo_source"]["model"] == "claude-opus-5"
    assert len(calls) == 1
    assert rule_based_memo(Toolbox(artifacts))["source"] == "rules"


def test_claude_failure_falls_back_to_rules(prepared, artifacts, monkeypatch):
    def failing_write_memo(toolbox):
        raise AgentError("rate limited")

    monkeypatch.setattr(server_module, "write_memo", failing_write_memo)
    app = MissionControl(
        paths=artifacts.paths,
        n_simulations=500,
        pace_seconds=0.0,
        use_claude=True,
        config=prepared.config,
        prepared=prepared,
    )
    response = app.stage("reporte")
    assert response["payload"]["recommendation"]["memo_source"]["author"] == "rules"
    assert not artifacts.paths.agent_memo.exists()


def test_served_payload_enables_the_api(server):
    base, _ = server
    payload = json.loads(request(f"{base}/api/payload")[2])["payload"]
    assert payload["mode"] == "server"
