import json
from types import SimpleNamespace

import pytest

from montecarlo_decisions import cli
from montecarlo_decisions.agent import claude


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_serve_defaults_are_safe():
    args = cli.build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.agent == "auto"


def test_dashboard_without_artifacts_fails_cleanly(tmp_path, capsys):
    assert cli.main(["dashboard", "--artifacts", str(tmp_path / "empty")]) == 2


def test_dashboard_from_artifacts(artifacts, tmp_path, capsys):
    code = cli.main(
        ["dashboard", "--artifacts", str(artifacts.paths.root), "--output", str(tmp_path)]
    )
    assert code == 0
    assert (tmp_path / "index.html").is_file()


def test_memo_without_credentials_fails_cleanly(artifacts, monkeypatch):
    class NoCredentials:
        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):  # what the SDK does when no credentials resolve
            raise TypeError("Could not resolve authentication method.")

    monkeypatch.setattr(claude.anthropic, "Anthropic", NoCredentials)
    assert cli.main(["memo", "--artifacts", str(artifacts.paths.root)]) == 2
    assert not artifacts.paths.agent_memo.exists()


def test_auto_agent_mode_follows_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not cli._has_api_credentials()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert cli._has_api_credentials()


@pytest.mark.slow
def test_pipeline_command_end_to_end(tmp_path, capsys):
    code = cli.main(
        [
            "pipeline",
            "--artifacts",
            str(tmp_path),
            "--simulations",
            "400",
            "--bootstrap-models",
            "3",
            "--jobs",
            "1",
        ]
    )
    checks = json.loads((tmp_path / "data" / "validation_checks.json").read_text(encoding="utf-8"))
    assert code == (0 if all(check["passed"] for check in checks) else 1)
    assert (tmp_path / "report.md").is_file()
    assert "Mejorar landing" in capsys.readouterr().out


def test_ask_prints_the_answer_and_trace(artifacts, monkeypatch, capsys):
    replies = [
        SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", id="c1", name="compare_decisions", input={})],
            stop_reason="tool_use",
            model="claude-opus-5",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        ),
        SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Prioriza el funnel.")],
            stop_reason="end_turn",
            model="claude-opus-5",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        ),
    ]

    class Scripted:
        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: replies.pop(0)))

    monkeypatch.setattr(claude.anthropic, "Anthropic", Scripted)
    assert cli.main(["ask", "Que priorizo?", "--artifacts", str(artifacts.paths.root)]) == 0
    captured = capsys.readouterr()
    assert "Prioriza el funnel." in captured.out
    assert "compare_decisions" in captured.err


def test_snapshot_copies_only_small_results(artifacts, tmp_path):
    output = tmp_path / "results"
    assert (
        cli.main(["snapshot", "--artifacts", str(artifacts.paths.root), "--output", str(output)])
        == 0
    )
    names = sorted(path.name for path in output.iterdir())
    assert names == sorted(
        [
            "report.md",
            "decision_summary.csv",
            "counterfactual_uplift.csv",
            "ads_budget_regimes.csv",
            "validation_checks.json",
        ]
    )
