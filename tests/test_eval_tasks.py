from pathlib import Path

import pytest

from harness.grader import load_task

TASKS = sorted((Path(__file__).resolve().parents[1] / "evals" / "tasks").glob("*.toml"))
KNOWN_CHECKS = {
    "just_check",
    "files_exist",
    "protected",
    "tests_preserved",
    "only_changed",
    "report_contains",
}


def test_there_are_tasks():
    assert TASKS


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.name)
def test_task_is_well_formed(path):
    task = load_task(path)
    assert task["id"].startswith(path.name.split("_")[0])
    assert task["prompt"].strip()
    assert task["checks"]


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.name)
def test_task_checks_are_known(path):
    assert set(load_task(path)["checks"]) <= KNOWN_CHECKS
