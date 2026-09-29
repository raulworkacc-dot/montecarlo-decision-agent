"""Local web app: Mission Control dashboard with a real, live Monte Carlo run.

Only an explicit allow-list of routes is served (no directory listing, no static file
serving from the project root), the server binds to localhost by default, and request
bodies are size-limited and validated.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from montecarlo_decisions.agent.claude import write_memo
from montecarlo_decisions.agent.memo import load_memo
from montecarlo_decisions.agent.tools import Toolbox
from montecarlo_decisions.artifacts import Artifacts, load_artifacts
from montecarlo_decisions.config import ArtifactPaths
from montecarlo_decisions.dashboard import (
    STAGES,
    build_payload,
    completed_status,
    render_live_dashboard,
    render_mission_control,
)
from montecarlo_decisions.live import STATUS_PREFIX, build_status, read_status, write_status
from montecarlo_decisions.pipeline import PreparedCase, prepare_case, run_simulation
from montecarlo_decisions.scenarios import ScenarioConfig, load_config

logger = logging.getLogger(__name__)
MAX_BODY_BYTES = 10_000
STAGE_KEYS = {stage["key"] for stage in STAGES}


@dataclass
class MissionControl:
    """Application state shared by all request threads."""

    paths: ArtifactPaths
    n_simulations: int
    pace_seconds: float
    use_claude: bool
    config: ScenarioConfig = field(default_factory=load_config)
    prepared: PreparedCase | None = None
    phase: str = "idle"  # idle | preparing | running | completed | failed
    error: str | None = None
    launch_requested: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)
    memo_lock: threading.Lock = field(default_factory=threading.Lock)

    # --- Lifecycle ----------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Make sure results exist before serving, and prepare models for live runs."""
        if not self.paths.is_complete():
            logger.info("No results yet: running the pipeline once (no pacing)")
            self._prepare()
            self._simulate(pace_seconds=0.0)
        else:
            write_status(self.paths.live_status, completed_status(self.artifacts()))
            threading.Thread(target=self._prepare, name="prepare", daemon=True).start()

    def _prepare(self) -> None:
        with self.lock:
            if self.prepared is not None or self.phase == "preparing":
                return
            self.phase = "preparing"
        try:
            prepared = prepare_case(self.config)
        except Exception as error:
            logger.exception("Preparation failed")
            with self.lock:
                self.phase, self.error = "failed", str(error)
            return
        with self.lock:
            self.prepared = prepared
            self.phase = "idle"
            launch = self.launch_requested
            self.launch_requested = False
        if launch:
            self.start_simulation()

    def _simulate(self, pace_seconds: float) -> None:
        with self.lock:
            prepared = self.prepared
            self.phase = "running"
        labels = self.config.labels()

        def progress(done: int, total: int, simulations: Any) -> None:
            write_status(
                self.paths.live_status,
                build_status(done, total, simulations, labels, message="Simulacion en curso."),
            )

        try:
            run_simulation(prepared, self.paths, self.n_simulations, progress, pace_seconds)
            write_status(self.paths.live_status, completed_status(self.artifacts()))
        except Exception as error:
            logger.exception("Simulation failed")
            with self.lock:
                self.phase, self.error = "failed", str(error)
            return
        with self.lock:
            self.phase = "completed"

    def start_simulation(self) -> str:
        """Launch a live run; returns the resulting phase."""
        with self.lock:
            if self.phase == "running":
                return "running"
            if self.prepared is None:
                self.launch_requested = True
                if self.phase != "preparing":
                    threading.Thread(target=self._prepare, name="prepare", daemon=True).start()
                return "preparing"
            self.phase = "running"
            self.error = None
        write_status(
            self.paths.live_status,
            build_status(0, self.n_simulations, pd.DataFrame(), {}, message="Lanzando simulacion."),
        )
        threading.Thread(
            target=self._simulate, args=(self.pace_seconds,), name="simulate", daemon=True
        ).start()
        return "running"

    # --- Views --------------------------------------------------------------------------

    def artifacts(self) -> Artifacts:
        return load_artifacts(self.paths, self.config)

    def memo(self) -> dict[str, Any] | None:
        """Saved memo for the current results, generating one with Claude if enabled."""
        artifacts = self.artifacts()
        saved = load_memo(self.paths.agent_memo, artifacts.fingerprint)
        if saved or not self.use_claude:
            return saved
        with self.memo_lock:
            saved = load_memo(self.paths.agent_memo, artifacts.fingerprint)
            if saved:
                return saved
            try:
                memo = write_memo(Toolbox(artifacts))
            except Exception as error:  # AgentError or any API/network failure -> rules
                logger.warning("Claude memo unavailable, using rule-based memo: %s", error)
                return None
            self.paths.agent_memo.write_text(
                json.dumps(memo, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return memo

    def payload(self, with_agent: bool = False) -> dict[str, Any]:
        memo = (
            self.memo()
            if with_agent
            else load_memo(self.paths.agent_memo, self.artifacts().fingerprint)
        )
        return build_payload(self.artifacts(), memo, self.pace_seconds, mode="server")

    def status(self) -> dict[str, Any]:
        with self.lock:
            phase, error = self.phase, self.error
        live = read_status(self.paths.live_status) or {}
        leaderboard = live.get("leaderboard") or []
        response = {
            "running": phase == "running",
            "is_running": phase == "running",
            "preparing": phase == "preparing" and self.launch_requested,
            "phase": phase,
            "error": error,
            "current_simulation": live.get("current_simulation", 0),
            "total_simulations": live.get("total_simulations", self.n_simulations),
            "progress_pct": live.get("progress_pct", 0.0),
            "status_message": error or live.get("status_message", ""),
            "leader": leaderboard[0] if leaderboard else None,
            "payload": None,
        }
        if phase == "completed":
            response["payload"] = self.payload()
        return response

    def stage(self, key: str) -> dict[str, Any]:
        stage = next(item for item in STAGES if item["key"] == key)
        if key == "montecarlo":
            phase = self.start_simulation()
            return {
                "stage": key,
                "status": phase,
                "delay_ms": 1,
                "payload": None,
                "logs": [
                    "> activando simulacion Monte Carlo",
                    f"> {self.n_simulations:,} futuros por decision".replace(",", "."),
                    "> precalculando modelos..."
                    if phase == "preparing"
                    else "> simulacion lanzada",
                ],
                "montecarlo": {"running": phase == "running", "preparing": phase == "preparing"},
            }
        payload = self.payload(with_agent=key == "reporte")
        logs = [f"> {stage['command']}", "> datos sincronizados con el backend"]
        if key == "reporte":
            source = payload["recommendation"]["memo_source"]
            author = f"Claude ({source['model']})" if source["author"] == "claude" else "reglas"
            logs += [
                f"> recomendacion: {payload['recommendation']['label']}",
                f"> memo redactado por: {author}",
            ]
        return {
            "stage": key,
            "status": "ok",
            "delay_ms": stage["duration_ms"],
            "payload": payload,
            "logs": logs,
        }


class Handler(BaseHTTPRequestHandler):
    app: MissionControl  # injected by make_server
    server_version = "MissionControl/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass  # the browser navigated away; nothing to do

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"status": "error", "error": message}, status)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path in {"/", "/index.html"}:
                html = render_mission_control(self.app.payload())
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/live.html":
                html = render_live_dashboard(self.app.config.labels())
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/live_status.js":
                status = read_status(self.app.paths.live_status) or completed_status(
                    self.app.artifacts()
                )
                body = STATUS_PREFIX + json.dumps(status, ensure_ascii=False) + ";"
                self._send(HTTPStatus.OK, body.encode("utf-8"), "text/javascript; charset=utf-8")
            elif path == "/api/payload":
                self._json({"status": "ok", "payload": self.app.payload()})
            elif path == "/api/montecarlo-status":
                self._json({"status": "ok", **self.app.status()})
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception:
            logger.exception("GET %s failed", path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal error")

    def _same_origin(self) -> bool:
        """Reject requests that a third-party page in the same browser sends to this app."""
        origin = self.headers.get("Origin")
        if origin is None:  # non-browser clients (curl, tests) send no Origin
            return True
        return urlsplit(origin).netloc == self.headers.get("Host", "")

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/api/stage":
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if not self._same_origin():
            self._error(HTTPStatus.FORBIDDEN, "Cross-origin request rejected")
            return
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            # also forces a CORS preflight on cross-site requests, which this server never grants
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Expected application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "Invalid body size")
            return
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return
        stage = data.get("stage") if isinstance(data, dict) else None
        if not isinstance(stage, str) or stage not in STAGE_KEYS:
            self._error(HTTPStatus.BAD_REQUEST, f"Unknown stage: {stage!r}")
            return
        try:
            self._json(self.app.stage(stage))
        except Exception:
            logger.exception("Stage %s failed", stage)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal error")


def make_server(app: MissionControl, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
