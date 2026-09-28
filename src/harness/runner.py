"""Runner de evals: ejecuta Claude Code sobre una copia limpia y puntua con el corrector.

Flujo: clona el repo (ultimo commit) en un directorio temporal, instala dependencias,
lanza `claude -p` con la tarea y puntua el resultado con harness.grader.

Los permisos de Bash se leen de .claude/settings.json y se pasan con --allowedTools,
porque Claude Code ignora `permissions.allow` en carpetas que no son de confianza.
No se usa --bare: omitiria CLAUDE.md, hooks y subagentes, es decir, el arnes.

Uso: uv run python -m harness.runner evals/tasks/001_add_utility.toml [--keep] [--timeout 600]
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
    "\n\nEste eval es no interactivo: nadie puede responder preguntas. "
    "Si algo es ambiguo, elige la opcion mas conservadora, documentala en el codigo "
    "y termina la tarea."
)


class RunnerError(Exception):
    """Error de preparacion del eval (no es un fallo del agente)."""


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
        raise RunnerError(f"no se encuentra el comando: {command[0]}") from error
    if result.returncode != 0:
        raise RunnerError(f"`{' '.join(command)}` fallo: {result.stderr.strip()}")
    return result.stdout


def ensure_clean(repo: Path) -> None:
    """La copia sale del ultimo commit: exige que no haya nada sin commitear."""
    if run_checked(["git", "status", "--porcelain"], repo).strip():
        raise RunnerError("hay cambios sin commitear: haz commit antes de lanzar evals")


def prepare_workdir(repo: Path, dest: Path, seed: bool = False) -> str:
    """Clona el repo en `dest`, instala dependencias y devuelve el sha base.

    Si `seed` es True ejecuta `just seed` (si tu proyecto tiene un paso de
    preparacion de datos/fixtures que debe correr una persona, no el agente,
    anadelo como esa receta). Sin receta `seed` en el justfile, una tarea que la
    pida fallara: no la uses hasta tener esa receta.
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
    """Extrae (resultado, coste estimado, id de sesion) del JSON de `claude -p`."""
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
    """En Windows los objetos de .git son de solo lectura: se fuerza el borrado."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup(workdir: Path, keep: bool) -> None:
    if keep:
        print(f"Copia conservada en: {workdir}")
    else:
        shutil.rmtree(workdir, onerror=_force_remove)


def run_task(task_path: Path, repo: Path, timeout: int, keep: bool = False) -> dict:
    task = load_task(task_path)
    ensure_clean(repo)
    allowed = load_allowed_tools(repo / ".claude" / "settings.json")
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        raise RunnerError("no se encuentra `claude` en el PATH")
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
        f"coste estimado {report['cost_usd']} USD)"
    )
    for check in report["checks"]:
        mark = "ok  " if check["passed"] else "FAIL"
        detail = f" - {check['detail']}" if check["detail"] else ""
        print(f"  {mark} {check['name']}{detail}")
    if report["timed_out"]:
        print("  El agente supero el tiempo limite.")
    print(f"Informe: {path}")


def discover_tasks(tasks_dir: Path) -> list[Path]:
    return sorted(tasks_dir.glob("*.toml"))


def format_summary(reports: list[dict]) -> str:
    rows = [("Tarea", "Resultado", "Tiempo", "Coste USD")]
    for report in reports:
        cost = report["cost_usd"]
        rows.append(
            (
                report["task_id"],
                "PASS" if report["passed"] else "FAIL",
                f"{report['duration_s']:.0f}s",
                "n/d" if cost is None else f"{cost:.3f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    passed = sum(1 for report in reports if report["passed"])
    lines.append(f"{passed}/{len(reports)} tareas pasan")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ejecuta tareas de referencia (evals).")
    parser.add_argument("task", type=Path, nargs="?", help="ruta al archivo .toml de la tarea")
    parser.add_argument(
        "--all", action="store_true", help="ejecuta todas las tareas de evals/tasks"
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="segundos por tarea")
    parser.add_argument("--keep", action="store_true", help="conserva la copia de trabajo")
    args = parser.parse_args(argv)
    if args.all == (args.task is not None):
        parser.error("indica una tarea o usa --all, pero no ambos")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.pop("VIRTUAL_ENV", None)  # evita el aviso de uv por el .venv del repo origen
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
