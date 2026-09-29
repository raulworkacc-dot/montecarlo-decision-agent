import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "block_protected_writes.py"


def write_config(project_dir: Path, paths: list[str]) -> None:
    config_dir = project_dir / ".claude"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "protected_paths.json").write_text(json.dumps(paths), encoding="utf-8")


def run_hook(project_dir: Path, stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)},
    )


def event(**tool_input: str) -> str:
    return json.dumps({"tool_name": "Write", "tool_input": tool_input})


def bash_event(command: str) -> str:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})


@pytest.mark.parametrize(
    "path",
    ["protected/x.csv", "protected/sub/x.csv", "protected/../protected/x.csv"],
)
def test_blocks_writes_in_protected_path(tmp_path, path):
    write_config(tmp_path, ["protected"])
    result = run_hook(tmp_path, event(file_path=path))
    assert result.returncode == 2


def test_blocks_absolute_path_in_protected(tmp_path):
    write_config(tmp_path, ["protected"])
    absolute = str(tmp_path / "protected" / "x.csv")
    assert run_hook(tmp_path, event(file_path=absolute)).returncode == 2


@pytest.mark.parametrize(
    "path",
    ["reports/x.csv", "src/harness/a.py", "protected_backup/x.csv"],
)
def test_allows_other_paths(tmp_path, path):
    write_config(tmp_path, ["protected"])
    assert run_hook(tmp_path, event(file_path=path)).returncode == 0


def test_allows_everything_when_no_config(tmp_path):
    assert run_hook(tmp_path, event(file_path="protected/x.csv")).returncode == 0


def test_allows_everything_when_config_is_empty_list(tmp_path):
    write_config(tmp_path, [])
    assert run_hook(tmp_path, event(file_path="protected/x.csv")).returncode == 0


def test_allows_events_without_file_path(tmp_path):
    write_config(tmp_path, ["protected"])
    assert run_hook(tmp_path, json.dumps({"tool_input": {"command": "ls"}})).returncode == 0


def test_blocks_unreadable_input(tmp_path):
    assert run_hook(tmp_path, "esto no es json").returncode == 2


def test_respects_multiple_protected_paths(tmp_path):
    write_config(tmp_path, ["protected", "secrets"])
    assert run_hook(tmp_path, event(file_path="secrets/key.pem")).returncode == 2
    assert run_hook(tmp_path, event(file_path="protected/x.csv")).returncode == 2
    assert run_hook(tmp_path, event(file_path="src/a.py")).returncode == 0


@pytest.mark.parametrize(
    "command",
    [
        "echo hola > protected/x.csv",
        "Set-Content protected\\x.csv 'a'",
        "cp foo.csv protected/foo.csv",
        "rm protected/x.csv",
        "python -c \"open('protected/x', 'w')\"",
        "cat protected/x.csv | tee protected/y.csv",
        "cat protected/x.csv > protected/y.csv",
        "Remove-Item PROTECTED\\x.csv",
    ],
)
def test_blocks_bash_touching_protected(tmp_path, command):
    write_config(tmp_path, ["protected"])
    assert run_hook(tmp_path, bash_event(command)).returncode == 2


@pytest.mark.parametrize(
    "command",
    [
        "ls protected",
        "Get-ChildItem protected",
        "cat protected/x.csv",
        "just check",
        "git status",
        "cp a.csv protected_backup/a.csv",
        "echo hola > reports/x.csv",
    ],
)
def test_allows_other_bash(tmp_path, command):
    write_config(tmp_path, ["protected"])
    assert run_hook(tmp_path, bash_event(command)).returncode == 0


@pytest.mark.parametrize(
    "command",
    [
        "ls protected\nrm protected/x.csv",
        "ls protected\r\nrm protected/x.csv",
        "cat protected/$(rm protected/x.csv)",
        "cat protected/`rm protected/x.csv`",
    ],
)
def test_blocks_bash_chained_after_read_only(tmp_path, command):
    write_config(tmp_path, ["protected"])
    assert run_hook(tmp_path, bash_event(command)).returncode == 2


def test_known_false_positive_pipe_is_blocked(tmp_path):
    # Accepted false positive: any pipe over a protected path is blocked.
    write_config(tmp_path, ["protected"])
    assert run_hook(tmp_path, bash_event("cat protected/x.csv | head")).returncode == 2


REPO = Path(__file__).resolve().parents[1]
ANALYSIS_INPUTS = [
    "src/montecarlo_decisions/synthetic.py",
    "src/montecarlo_decisions/scenarios.toml",
]


def test_project_protects_the_analysis_inputs():
    config = REPO / ".claude" / "protected_paths.json"
    assert set(ANALYSIS_INPUTS) <= set(json.loads(config.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", ANALYSIS_INPUTS)
def test_real_config_blocks_edits_to_analysis_inputs(path):
    result = run_hook(REPO, event(file_path=path))
    assert result.returncode == 2
    assert "protected" in result.stderr


def test_real_config_allows_other_source_files():
    assert run_hook(REPO, event(file_path="src/montecarlo_decisions/models.py")).returncode == 0
