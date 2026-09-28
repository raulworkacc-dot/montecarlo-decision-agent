"""Corrector determinista para las tareas de referencia (evals).

No usa ningun modelo: solo git, el sistema de archivos y el codigo de salida de
`just check`. Recibe el repo donde trabajo el agente y el sha del commit base desde
el que se lanzo la tarea.

Limite conocido: los archivos ignorados por .gitignore no se detectan.
"""

import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECK_COMMAND = ("just", "check")


@dataclass(frozen=True)
class Result:
    name: str
    passed: bool
    detail: str = ""


def load_task(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _run(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git(repo: Path, *args: str) -> str:
    result = _run(["git", *args], repo)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} fallo: {result.stderr.strip()}")
    return result.stdout


def changed_paths(repo: Path, base: str, paths: Sequence[str]) -> list[str]:
    """Rutas bajo `paths` que difieren del commit base (nuevas, editadas o borradas)."""
    if not paths:
        return []
    tracked = _git(repo, "diff", "--name-only", "--no-renames", base, "--", *paths)
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", *paths)
    return sorted({line for line in (tracked + untracked).splitlines() if line})


def check_files_exist(repo: Path, files: Sequence[str]) -> Result:
    missing = [name for name in files if not (repo / name).is_file()]
    return Result("files_exist", not missing, f"faltan: {missing}" if missing else "")


def check_protected(repo: Path, base: str, protected: Sequence[str]) -> Result:
    changed = changed_paths(repo, base, protected)
    return Result("protected", not changed, f"alterado: {changed}" if changed else "")


def check_tests_preserved(repo: Path, base: str) -> Result:
    """Se pueden anadir tests, pero no modificar ni borrar los que ya existian."""
    out = _git(repo, "diff", "--name-status", "--no-renames", base, "--", "tests")
    touched = [line for line in out.splitlines() if line and not line.startswith("A")]
    return Result("tests_preserved", not touched, f"tests alterados: {touched}" if touched else "")


def check_only_changed(repo: Path, base: str, allowed: Sequence[str]) -> Result:
    """Todo cambio respecto al commit base debe estar dentro de las rutas permitidas."""
    prefixes = [item.rstrip("/") for item in allowed]
    tracked = _git(repo, "diff", "--name-only", "--no-renames", base)
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard")
    changed = {line for line in (tracked + untracked).splitlines() if line}
    outside = sorted(
        path
        for path in changed
        if not any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)
    )
    detail = f"fuera de lo permitido: {outside}" if outside else ""
    return Result("only_changed", not outside, detail)


def check_report_contains(repo: Path, spec: dict) -> Result:
    """El informe existe y menciona cada termino (sin distinguir mayusculas).

    Cada termino admite alternativas separadas por `|`: basta con que aparezca una.
    Es una comprobacion superficial: no verifica que el informe lo explique bien.
    """
    name = spec["path"]
    path = repo / name
    if not path.is_file():
        return Result("report_contains", False, f"no existe: {name}")
    terms = spec.get("terms", [])
    if not terms:
        return Result("report_contains", False, "la tarea no define terminos")
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    missing = []
    for term in terms:
        options = [option.strip().lower() for option in term.split("|") if option.strip()]
        if not any(option in text for option in options):
            missing.append(term)
    return Result("report_contains", not missing, f"faltan: {missing}" if missing else "")


def check_just(repo: Path, command: Sequence[str] = DEFAULT_CHECK_COMMAND) -> Result:
    try:
        run = _run(command, repo)
    except FileNotFoundError:
        return Result("just_check", False, f"comando no encontrado: {command[0]}")
    passed = run.returncode == 0
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-5:])
    return Result("just_check", passed, "" if passed else tail)


def grade(
    task: dict,
    repo: Path,
    base: str,
    check_command: Sequence[str] = DEFAULT_CHECK_COMMAND,
) -> list[Result]:
    checks = task.get("checks", {})
    results: list[Result] = []
    if checks.get("just_check"):
        results.append(check_just(repo, check_command))
    if "files_exist" in checks:
        results.append(check_files_exist(repo, checks["files_exist"]))
    if "protected" in checks:
        results.append(check_protected(repo, base, checks["protected"]))
    if checks.get("tests_preserved"):
        results.append(check_tests_preserved(repo, base))
    if "only_changed" in checks:
        results.append(check_only_changed(repo, base, checks["only_changed"]))
    if "report_contains" in checks:
        results.append(check_report_contains(repo, checks["report_contains"]))
    return results


def all_passed(results: Sequence[Result]) -> bool:
    return bool(results) and all(result.passed for result in results)
