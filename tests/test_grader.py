import subprocess
import sys
from pathlib import Path

import pytest

from harness.grader import (
    all_passed,
    check_files_exist,
    check_just,
    check_only_changed,
    check_protected,
    check_report_contains,
    check_tests_preserved,
    grade,
    load_task,
)

ROOT = Path(__file__).resolve().parents[1]
OK_CMD = (sys.executable, "-c", "raise SystemExit(0)")
FAIL_CMD = (sys.executable, "-c", "raise SystemExit(1)")
ALL_CHECKS = {
    "just_check": True,
    "files_exist": ["pyproject.toml"],
    "protected": ["protected"],
    "tests_preserved": True,
}


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "tests").mkdir()
    write(tmp_path / "tests" / "test_a.py", "def test_a():\n    assert True\n")
    (tmp_path / "protected").mkdir()
    write(tmp_path / "protected" / "seed.csv", "a,b\n1,2\n")
    write(tmp_path / "pyproject.toml", "[project]\nname = 'x'\n")
    git(tmp_path, "init")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "base")
    return tmp_path


@pytest.fixture
def base(repo):
    return git(repo, "rev-parse", "HEAD")


def test_files_exist_ok(repo):
    assert check_files_exist(repo, ["pyproject.toml"]).passed


def test_files_exist_reports_missing(repo):
    result = check_files_exist(repo, ["pyproject.toml", "src/nuevo.py"])
    assert not result.passed
    assert "src/nuevo.py" in result.detail


def test_protected_ignores_unrelated_changes(repo, base):
    (repo / "src").mkdir()
    write(repo / "src" / "a.py", "x = 1\n")
    assert check_protected(repo, base, ["protected", "pyproject.toml"]).passed


def test_protected_detects_modified_file(repo, base):
    write(repo / "pyproject.toml", "[project]\nname = 'y'\n")
    result = check_protected(repo, base, ["protected", "pyproject.toml"])
    assert not result.passed
    assert "pyproject.toml" in result.detail


def test_protected_detects_new_file_in_protected_dir(repo, base):
    write(repo / "protected" / "nuevo.csv", "x\n")
    assert not check_protected(repo, base, ["protected"]).passed


def test_protected_detects_deleted_file(repo, base):
    (repo / "protected" / "seed.csv").unlink()
    assert not check_protected(repo, base, ["protected"]).passed


def test_protected_detects_committed_change(repo, base):
    write(repo / "protected" / "seed.csv", "a,b\n9,9\n")
    git(repo, "commit", "-am", "el agente commitea")
    assert not check_protected(repo, base, ["protected"]).passed


def test_tests_preserved_allows_new_tests(repo, base):
    write(repo / "tests" / "test_b.py", "def test_b():\n    assert True\n")
    assert check_tests_preserved(repo, base).passed


def test_tests_preserved_detects_modification(repo, base):
    write(repo / "tests" / "test_a.py", "def test_a():\n    pass\n")
    assert not check_tests_preserved(repo, base).passed


def test_tests_preserved_detects_deletion(repo, base):
    (repo / "tests" / "test_a.py").unlink()
    assert not check_tests_preserved(repo, base).passed


def test_tests_preserved_detects_committed_deletion(repo, base):
    (repo / "tests" / "test_a.py").unlink()
    git(repo, "commit", "-am", "el agente borra un test y commitea")
    assert not check_tests_preserved(repo, base).passed


def test_just_check_passes_on_exit_zero(repo):
    assert check_just(repo, OK_CMD).passed


def test_just_check_fails_on_nonzero_exit(repo):
    assert not check_just(repo, FAIL_CMD).passed


def test_just_check_reports_missing_command(repo):
    result = check_just(repo, ("comando-que-no-existe-xyz",))
    assert not result.passed
    assert "not found" in result.detail


def test_grade_runs_only_declared_checks(repo, base):
    task = {"checks": {"files_exist": ["pyproject.toml"]}}
    results = grade(task, repo, base, OK_CMD)
    assert [r.name for r in results] == ["files_exist"]


def test_grade_all_checks_pass(repo, base):
    results = grade({"checks": ALL_CHECKS}, repo, base, OK_CMD)
    names = [r.name for r in results]
    assert names == ["just_check", "files_exist", "protected", "tests_preserved"]
    assert all_passed(results)


def test_grade_fails_if_any_check_fails(repo, base):
    results = grade({"checks": ALL_CHECKS}, repo, base, FAIL_CMD)
    assert not all_passed(results)


def test_load_task_reads_reference_task():
    task = load_task(ROOT / "evals" / "tasks" / "001_add_utility.toml")
    assert task["id"] == "001-add-utility"
    assert task["checks"]["just_check"] is True


def test_only_changed_allows_new_files_in_allowed_dir(repo, base):
    (repo / "reports").mkdir()
    write(repo / "reports" / "salida.md", "hola\n")
    assert check_only_changed(repo, base, ["reports/"]).passed


def test_only_changed_detects_modified_file_outside(repo, base):
    write(repo / "pyproject.toml", "[project]\nname = 'y'\n")
    result = check_only_changed(repo, base, ["reports/"])
    assert not result.passed
    assert "pyproject.toml" in result.detail


def test_only_changed_detects_new_file_outside(repo, base):
    (repo / "src").mkdir()
    write(repo / "src" / "extra.py", "x = 1\n")
    assert not check_only_changed(repo, base, ["reports"]).passed


def test_only_changed_does_not_match_prefix_lookalikes(repo, base):
    (repo / "reports_old").mkdir()
    write(repo / "reports_old" / "x.md", "x\n")
    assert not check_only_changed(repo, base, ["reports"]).passed


def test_only_changed_ignores_gitignored_files(repo):
    write(repo / ".gitignore", "cache.bin\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "ignora el cache")
    new_base = git(repo, "rev-parse", "HEAD")
    write(repo / "cache.bin", "binario\n")
    assert check_only_changed(repo, new_base, ["reports"]).passed


def test_report_contains_accepts_alternatives_case_insensitive(repo):
    (repo / "reports").mkdir()
    write(repo / "reports" / "informe.md", "Se encontro un problema. Cifra: 999.\n")
    spec = {"path": "reports/informe.md", "terms": ["problema|issue", "999"]}
    assert check_report_contains(repo, spec).passed


def test_report_contains_lists_missing_terms(repo):
    (repo / "reports").mkdir()
    write(repo / "reports" / "informe.md", "solo habla de un problema\n")
    spec = {"path": "reports/informe.md", "terms": ["problema", "fuga|leak", "999"]}
    result = check_report_contains(repo, spec)
    assert not result.passed
    assert "fuga|leak" in result.detail
    assert "999" in result.detail
    assert "problema" not in result.detail


def test_report_contains_fails_when_file_missing(repo):
    spec = {"path": "reports/informe.md", "terms": ["x"]}
    assert not check_report_contains(repo, spec).passed


def test_report_contains_fails_without_terms(repo):
    (repo / "reports").mkdir()
    write(repo / "reports" / "informe.md", "algo\n")
    assert not check_report_contains(repo, {"path": "reports/informe.md", "terms": []}).passed


def test_grade_runs_content_checks(repo, base):
    (repo / "reports").mkdir()
    write(repo / "reports" / "informe.md", "se encontro un problema\n")
    checks = {
        "only_changed": ["reports/"],
        "report_contains": {"path": "reports/informe.md", "terms": ["problema"]},
    }
    results = grade({"checks": checks}, repo, base, OK_CMD)
    assert [r.name for r in results] == ["only_changed", "report_contains"]
    assert all_passed(results)
