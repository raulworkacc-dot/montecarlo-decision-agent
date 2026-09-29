"""Command-line interface: ``mcd <command>``.

mcd pipeline      generate data, fit models, simulate, validate, write artifacts
mcd dashboard     build the static Mission Control dashboard from the artifacts
mcd snapshot      copy the small, reviewable results into docs/results
mcd serve         run the local web app with a live Monte Carlo
mcd memo          have Claude write the decision memo (needs ANTHROPIC_API_KEY)
mcd ask "..."     ask the Claude agent a free-form question about the results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from montecarlo_decisions import __version__
from montecarlo_decisions.config import SEED, artifact_paths

logger = logging.getLogger("montecarlo_decisions")


def _add_artifacts_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="artifact directory (default: $MCD_ARTIFACTS_DIR or ./artifacts)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcd", description="Monte Carlo decision agent: counterfactuals, risk and a memo."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    pipeline = commands.add_parser("pipeline", help="run the full analysis and write artifacts")
    _add_artifacts_option(pipeline)
    pipeline.add_argument("--simulations", type=int, default=10_000, help="futures per decision")
    pipeline.add_argument("--seed", type=int, default=SEED)
    pipeline.add_argument(
        "--bootstrap-models", type=int, default=None, help="override n_bootstrap_models"
    )
    pipeline.add_argument("--jobs", type=int, default=-1, help="parallel workers (joblib)")

    dashboard = commands.add_parser("dashboard", help="build the static dashboard")
    _add_artifacts_option(dashboard)
    dashboard.add_argument(
        "--output", type=Path, default=None, help="output folder (default: <artifacts>/dashboards)"
    )

    snapshot = commands.add_parser("snapshot", help="copy small result files for versioning")
    _add_artifacts_option(snapshot)
    snapshot.add_argument("--output", type=Path, default=Path("docs") / "results")

    serve = commands.add_parser("serve", help="run the local Mission Control web app")
    _add_artifacts_option(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--simulations", type=int, default=10_000)
    serve.add_argument(
        "--pace-seconds",
        type=float,
        default=20.0,
        help="stretch the live run to ~N seconds so it can be followed (0 = full speed)",
    )
    serve.add_argument(
        "--agent",
        choices=["auto", "claude", "rules"],
        default="auto",
        help="memo author: auto = Claude when ANTHROPIC_API_KEY is set, else rules",
    )
    serve.add_argument("--no-browser", action="store_true")

    memo = commands.add_parser("memo", help="write the decision memo with Claude")
    _add_artifacts_option(memo)
    memo.add_argument("--model", default=None)

    ask = commands.add_parser("ask", help="ask the Claude agent a question")
    _add_artifacts_option(ask)
    ask.add_argument("question")
    ask.add_argument("--model", default=None)
    return parser


def cmd_pipeline(args: argparse.Namespace) -> int:
    from montecarlo_decisions.pipeline import run_pipeline
    from montecarlo_decisions.report import decision_table, markdown_table

    paths = artifact_paths(args.artifacts)
    result = run_pipeline(
        paths,
        n_simulations=args.simulations,
        seed=args.seed,
        n_bootstrap_models=args.bootstrap_models,
        n_jobs=args.jobs,
    )
    print(markdown_table(decision_table(result.summary)))
    print()
    for check in result.checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print(f"\nArtifacts: {paths.root}\nReport:    {paths.report}")
    return 0 if result.passed else 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    from montecarlo_decisions.artifacts import load_artifacts
    from montecarlo_decisions.dashboard import build_static_site

    paths = artifact_paths(args.artifacts)
    index = build_static_site(load_artifacts(paths), args.output or paths.dashboards_dir)
    print(f"Dashboard: {index}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    import shutil

    from montecarlo_decisions.artifacts import ArtifactsMissingError

    paths = artifact_paths(args.artifacts)
    if not paths.is_complete():
        raise ArtifactsMissingError(f"No complete results in {paths.root}. Run `mcd pipeline`.")
    args.output.mkdir(parents=True, exist_ok=True)
    for source in (paths.report, paths.summary, paths.uplift, paths.ads_regimes, paths.checks):
        shutil.copy2(source, args.output / source.name)
    print(f"Snapshot: {args.output}")
    return 0


def _has_api_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def cmd_serve(args: argparse.Namespace) -> int:
    from montecarlo_decisions.server import MissionControl, make_server

    use_claude = args.agent == "claude" or (args.agent == "auto" and _has_api_credentials())
    app = MissionControl(
        paths=artifact_paths(args.artifacts),
        n_simulations=args.simulations,
        pace_seconds=args.pace_seconds,
        use_claude=use_claude,
    )
    app.bootstrap()
    server = make_server(app, args.host, args.port)
    url = f"http://{args.host}:{server.server_port}/"
    if args.host not in {"127.0.0.1", "localhost"}:
        logger.warning("Listening on %s: the app has no authentication.", args.host)
    print(f"Mission Control: {url}  (memo: {'Claude' if use_claude else 'rules'}; Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_memo(args: argparse.Namespace) -> int:
    from montecarlo_decisions.agent.claude import DEFAULT_MODEL, write_memo
    from montecarlo_decisions.agent.tools import Toolbox
    from montecarlo_decisions.artifacts import load_artifacts

    artifacts = load_artifacts(artifact_paths(args.artifacts))
    memo = write_memo(Toolbox(artifacts), model=args.model or DEFAULT_MODEL)
    artifacts.paths.agent_memo.write_text(
        json.dumps(memo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{memo['headline']}\n\n{memo['summary']}\n")
    print(f"Tool calls: {len(memo['tool_trace'])}. Saved to {artifacts.paths.agent_memo}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from montecarlo_decisions.agent.claude import DEFAULT_MODEL, ask
    from montecarlo_decisions.agent.tools import Toolbox
    from montecarlo_decisions.artifacts import load_artifacts

    toolbox = Toolbox(load_artifacts(artifact_paths(args.artifacts)))
    run = ask(toolbox, args.question, model=args.model or DEFAULT_MODEL)
    for step in run.trace:
        print(f"  -> {step['name']}({step['args']}): {step['outcome']}", file=sys.stderr)
    print(run.answer)
    return 0


COMMANDS = {
    "pipeline": cmd_pipeline,
    "dashboard": cmd_dashboard,
    "snapshot": cmd_snapshot,
    "serve": cmd_serve,
    "memo": cmd_memo,
    "ask": cmd_ask,
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    import anthropic

    from montecarlo_decisions.agent.claude import AgentError
    from montecarlo_decisions.artifacts import ArtifactsMissingError

    try:
        return COMMANDS[args.command](args)
    except (ArtifactsMissingError, AgentError) as error:
        logger.error("%s", error)
        return 2
    except anthropic.AnthropicError as error:
        logger.error("Claude API unavailable (%s). Set ANTHROPIC_API_KEY in .env.", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
