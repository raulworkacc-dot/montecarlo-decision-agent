"""Eval runner: runs Claude Code on a clean copy of the repo and scores it with the grader.

Flow: clone the repo (last commit) into a temporary directory, install dependencies,
launch `claude -p` with the task and score the result with harness.grader.

Bash permissions are read from .claude/settings.json and passed with --allowedTools,
because Claude Code ignores `permissions.allow` in untrusted folders. --bare is not
used: it would skip CLAUDE.md, hooks and subagents, i.e. the harness itself.

Usage: uv run python -m harness.runner evals/tasks/001_add_utility.toml [--keep] [--timeout 600]
"""

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from harness.grader import all_passed, grade, load_task

DEFAULT_TIMEOUT = 600

NON_INTERACTIVE_NOTE = (
    "\n\nThis eval is non-interactive: nobody can answer questions. "
    "If something is ambiguous, choose the most conservative option, document it in "
    "the code and finish the task."
)


class RunnerError(Exception):
    """Eval setup error (not an agent failure)."""


@dataclass
class AgentRun:
    returncode: int | None
    timed_out: bool
    duration_s: float
    result_text: str = ""
    cost_usd: float | None = None
    session_id: str | None = None
    output_tail: str = ""


def run_checked(command: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as error:
        raise RunnerError(f"command not found: {command[0]}") from error
    if result.returncode != 0:
        raise RunnerError(f"`{' '.join(command)}` failed: {result.stderr.strip()}")
    return result.stdout


def ensure_clean(repo: Path) -> None:
    """The copy comes from the last commit: require a clean working tree."""
    if run_checked(["git", "status", "--porcelain"], repo).strip():
        raise RunnerError("uncommitted changes: commit before running evals")


def prepare_workdir(repo: Path, dest: Path, seed: bool = False) -> str:
    """Clone the repo into `dest`, install dependencies and return the base sha.

    If `seed` is True it runs `just seed` (a data/fixture preparation step that a
    person, not the agent, should run). Without a `seed` recipe in the justfile a task
    that asks for it fails.
    """
    run_checked(["git", "clone", "--quiet", str(repo), str(dest)], repo)
    run_checked(["uv", "sync", "--quiet"], dest)
    if seed:
        run_checked(["just", "seed"], dest)
    return run_checked(["git", "rev-parse", "HEAD"], dest).strip()


def load_allowed_tools(settings_path: Path) -> list[str]:
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    return list(data.get("permissions", {}).get("allow", []))


def build_prompt(task_prompt: str) -> str:
    return task_prompt.strip() + NON_INTERACTIVE_NOTE


def build_command(prompt: str, allowed_tools: list[str], claude_bin: str = "claude") -> list[str]:
    command = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
        "--permission-prompts",
        "none",
    ]
    if allowed_tools:
        command += ["--allowedTools", ",".join(allowed_tools)]
    return command


def parse_output(stdout: str) -> tuple[str, float | None, str | None]:
    """Extract (result, estimated cost, session id) from the JSON of `claude -p`."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return "", None, None
    if not isinstance(data, dict):
        return "", None, None
    return str(data.get("result", "")), data.get("total_cost_usd"), data.get("session_id")


def run_agent(command: list[str], cwd: Path, timeout: int) -> AgentRun:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return AgentRun(returncode=None, timed_out=True, duration_s=time.monotonic() - start)
    text, cost, session_id = parse_output(proc.stdout)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
    return AgentRun(
        returncode=proc.returncode,
        timed_out=False,
        duration_s=time.monotonic() - start,
        result_text=text,
        cost_usd=cost,
        session_id=session_id,
        output_tail=tail,
    )


def _force_remove(func, path, _exc_info) -> None:
    """On Windows .git objects are read-only: force the removal."""
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def cleanup(workdir: Path, keep: bool) -> None:
    if keep:
        print(f"Working copy kept at: {workdir}")
    else:
        shutil.rmtree(workdir, onerror=_force_remove)


def run_task(task_path: Path, repo: Path, timeout: int, keep: bool = False) -> dict:
    task = load_task(task_path)
    ensure_clean(repo)
    allowed = load_allowed_tools(repo / ".claude" / "settings.json")
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        raise RunnerError("`claude` is not on the PATH")
    workdir = Path(tempfile.mkdtemp(prefix="harness-eval-"))
    try:
        dest = workdir / "repo"
        base = prepare_workdir(repo, dest, seed=bool(task.get("seed", False)))
        prompt = build_prompt(task["prompt"])
        agent = run_agent(build_command(prompt, allowed, claude_bin), dest, timeout)
        results = grade(task, dest, base)
    finally:
        cleanup(workdir, keep)
    return {
        "task_id": task["id"],
        "passed": all_passed(results) and not agent.timed_out,
        "timed_out": agent.timed_out,
        "returncode": agent.returncode,
        "duration_s": round(agent.duration_s, 1),
        "cost_usd": agent.cost_usd,
        "session_id": agent.session_id,
        "agent_result": agent.result_text,
        "agent_output_tail": agent.output_tail,
        "base_sha": base,
        "checks": [asdict(result) for result in results],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_report(report: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = results_dir / f"{report['task_id']}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_report(report: dict, path: Path) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"[{status}] {report['task_id']} ({report['duration_s']:.0f}s, "
        f"estimated cost {report['cost_usd']} USD)"
    )
    for check in report["checks"]:
        mark = "ok  " if check["passed"] else "FAIL"
        detail = f" - {check['detail']}" if check["detail"] else ""
        print(f"  {mark} {check['name']}{detail}")
    if report["timed_out"]:
        print("  The agent exceeded the time limit.")
    print(f"Report: {path}")


def discover_tasks(tasks_dir: Path) -> list[Path]:
    return sorted(tasks_dir.glob("*.toml"))


def format_summary(reports: list[dict]) -> str:
    rows = [("Task", "Result", "Time", "Cost USD")]
    for report in reports:
        cost = report["cost_usd"]
        rows.append(
            (
                report["task_id"],
                "PASS" if report["passed"] else "FAIL",
                f"{report['duration_s']:.0f}s",
                "n/a" if cost is None else f"{cost:.3f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    passed = sum(1 for report in reports if report["passed"])
    lines.append(f"{passed}/{len(reports)} tasks pass")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the harness reference tasks (evals).")
    parser.add_argument("task", type=Path, nargs="?", help="path to the task .toml file")
    parser.add_argument("--all", action="store_true", help="run every task in evals/tasks")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per task")
    parser.add_argument("--keep", action="store_true", help="keep the working copy")
    args = parser.parse_args(argv)
    if args.all == (args.task is not None):
        parser.error("pass a task or --all, not both")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.pop("VIRTUAL_ENV", None)  # avoid uv's warning about the source repo's .venv
    repo = Path(__file__).resolve().parents[2]
    tasks = discover_tasks(repo / "evals" / "tasks") if args.all else [args.task.resolve()]
    reports = []
    for task_path in tasks:
        try:
            report = run_task(task_path, repo, args.timeout, args.keep)
        except RunnerError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        path = save_report(report, repo / "evals" / "results")
        print_report(report, path)
        reports.append(report)
    if args.all:
        print()
        print(format_summary(reports))
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
