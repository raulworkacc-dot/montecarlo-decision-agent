import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness.runner import (
    RunnerError,
    build_command,
    build_prompt,
    discover_tasks,
    ensure_clean,
    format_summary,
    load_allowed_tools,
    main,
    parse_output,
    prepare_workdir,
    run_agent,
    save_report,
)

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> None:
    subprocess.run(
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


@pytest.fixture
def committed_repo(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    git(tmp_path, "init")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "base")
    return tmp_path


def test_load_allowed_tools_reads_permissions_allow(tmp_path):
    settings = tmp_path / "settings.json"
    data = {"permissions": {"allow": ["Bash(just *)", "Bash(git status)"]}}
    settings.write_text(json.dumps(data), encoding="utf-8")
    assert load_allowed_tools(settings) == ["Bash(just *)", "Bash(git status)"]


def test_load_allowed_tools_missing_section_returns_empty(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    assert load_allowed_tools(settings) == []


def test_real_settings_allow_only_specific_just_recipes():
    allowed = load_allowed_tools(ROOT / ".claude" / "settings.json")
    assert "Bash(just check)" in allowed
    assert "Bash(just *)" not in allowed
    assert not any("seed" in rule or "eval" in rule for rule in allowed)


def test_build_command_prompt_follows_dash_p():
    command = build_command("haz algo", ["Bash(just *)"], "claude")
    assert command[:3] == ["claude", "-p", "haz algo"]


def test_build_command_passes_allowed_tools_as_single_argument():
    command = build_command("x", ["Bash(just *)", "Bash(git status)"])
    index = command.index("--allowedTools")
    assert command[index + 1] == "Bash(just *),Bash(git status)"


def test_build_command_flags():
    command = build_command("x", [])
    assert "--allowedTools" not in command
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    assert command[command.index("--permission-prompts") + 1] == "none"
    assert "--bare" not in command


def test_parse_output_reads_fields():
    payload = json.dumps({"result": "listo", "total_cost_usd": 0.05, "session_id": "abc"})
    assert parse_output(payload) == ("listo", 0.05, "abc")


@pytest.mark.parametrize("raw", ["", "no es json", "[1, 2]"])
def test_parse_output_tolerates_garbage(raw):
    assert parse_output(raw) == ("", None, None)


def test_ensure_clean_accepts_clean_repo(committed_repo):
    ensure_clean(committed_repo)


def test_ensure_clean_rejects_uncommitted_changes(committed_repo):
    (committed_repo / "b.txt").write_text("y", encoding="utf-8")
    with pytest.raises(RunnerError):
        ensure_clean(committed_repo)


def test_run_agent_parses_json_output(tmp_path):
    payload = json.dumps({"result": "ok", "total_cost_usd": 0.5, "session_id": "s1"})
    run = run_agent([sys.executable, "-c", f"print({payload!r})"], tmp_path, timeout=30)
    assert (run.returncode, run.timed_out) == (0, False)
    assert (run.result_text, run.cost_usd, run.session_id) == ("ok", 0.5, "s1")


def test_run_agent_reports_timeout(tmp_path):
    run = run_agent([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, timeout=1)
    assert run.timed_out
    assert run.returncode is None


def test_save_report_writes_json_with_task_id(tmp_path):
    report = {"task_id": "001-x", "passed": True}
    path = save_report(report, tmp_path / "results")
    assert path.name.startswith("001-x-")
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_build_prompt_appends_non_interactive_note():
    prompt = build_prompt("  haz algo  ")
    assert prompt.startswith("haz algo")
    assert "no interactivo" in prompt
    assert "nadie puede responder" in prompt


def test_format_summary_lists_tasks_and_counts():
    reports = [
        {"task_id": "001-a", "passed": True, "duration_s": 55.4, "cost_usd": 0.12415},
        {"task_id": "002-b", "passed": False, "duration_s": 64.0, "cost_usd": None},
    ]
    text = format_summary(reports)
    assert "001-a" in text and "PASS" in text
    assert "002-b" in text and "FAIL" in text
    assert "0.124" in text
    assert "n/d" in text
    assert text.splitlines()[-1] == "1/2 tareas pasan"


def test_discover_tasks_returns_sorted_toml_files(tmp_path):
    for name in ["b.toml", "a.toml", "notes.txt"]:
        (tmp_path / name).write_text("", encoding="utf-8")
    assert [path.name for path in discover_tasks(tmp_path)] == ["a.toml", "b.toml"]


@pytest.mark.parametrize("argv", [[], ["x.toml", "--all"]])
def test_main_requires_exactly_one_of_task_or_all(argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_prepare_workdir_seeds_when_requested(monkeypatch, tmp_path):
    commands = []

    def fake_run_checked(command, cwd):
        commands.append(command)
        return "abc123\n"

    monkeypatch.setattr("harness.runner.run_checked", fake_run_checked)
    sha = prepare_workdir(tmp_path, tmp_path / "copia", seed=True)
    assert sha == "abc123"
    assert ["just", "seed"] in commands


def test_prepare_workdir_does_not_seed_by_default(monkeypatch, tmp_path):
    commands = []

    def fake_run_checked(command, cwd):
        commands.append(command)
        return "abc123\n"

    monkeypatch.setattr("harness.runner.run_checked", fake_run_checked)
    prepare_workdir(tmp_path, tmp_path / "copia")
    assert ["just", "seed"] not in commands
