"""The agent loop, driven by a scripted fake client (no network, no API key)."""

from types import SimpleNamespace

import pytest

from montecarlo_decisions.agent import claude
from montecarlo_decisions.agent.memo import MEMO_SCHEMA, rule_based_memo
from montecarlo_decisions.agent.tools import Toolbox


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(name: str, arguments: dict, call_id: str = "call") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def response(*content, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        model="claude-opus-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class FakeClient:
    """Returns scripted responses and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture(scope="module")
def toolbox(artifacts) -> Toolbox:
    return Toolbox(artifacts)


@pytest.fixture(scope="module")
def memo_body(toolbox) -> dict:
    memo = rule_based_memo(toolbox)
    return {key: memo[key] for key in MEMO_SCHEMA["required"]}


def test_ask_runs_tools_then_answers(toolbox):
    client = FakeClient(
        response(tool_use("compare_decisions", {}, "c1")),
        response(text("Recomiendo el funnel."), stop_reason="end_turn"),
    )
    run = claude.ask(toolbox, "Que hago?", client=client)
    assert run.answer == "Recomiendo el funnel."
    assert run.turns == 2
    assert [step["name"] for step in run.trace] == ["compare_decisions"]
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "c1"
    assert "is_error" not in tool_result


def test_requests_use_fallbacks_thinking_and_strict_tools(toolbox):
    client = FakeClient(response(text("ok"), stop_reason="end_turn"))
    claude.ask(toolbox, "hola", client=client)
    request = client.requests[0]
    assert request["model"] == claude.DEFAULT_MODEL
    assert request["fallbacks"] == "default"
    assert claude.FALLBACK_BETA in request["betas"]
    assert request["thinking"] == {"type": "adaptive"}
    assert all(tool["strict"] for tool in request["tools"])


def test_invalid_tool_input_is_reported_to_the_model(toolbox):
    client = FakeClient(
        response(tool_use("get_lever_uplift", {"lever": "bitcoin"}, "bad")),
        response(text("Entendido."), stop_reason="end_turn"),
    )
    run = claude.ask(toolbox, "?", client=client)
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "not one of" in tool_result["content"]
    assert run.trace[0]["outcome"].startswith("Error")


def test_write_memo_returns_a_validated_claude_memo(toolbox, memo_body):
    client = FakeClient(
        response(tool_use("get_validation_checks", {}, "c1")),
        response(tool_use(claude.SUBMIT_MEMO_TOOL, memo_body, "c2")),
    )
    memo = claude.write_memo(toolbox, client=client)
    assert memo["source"] == "claude"
    assert memo["model"] == "claude-opus-5"
    assert memo["fingerprint"] == toolbox.artifacts.fingerprint
    assert [step["name"] for step in memo["tool_trace"]] == ["get_validation_checks"]
    tool_names = [tool["name"] for tool in client.requests[0]["tools"]]
    assert claude.SUBMIT_MEMO_TOOL in tool_names


def test_invalid_memo_raises(toolbox, memo_body):
    broken = {**memo_body, "recommended_decision": "lottery"}
    client = FakeClient(response(tool_use(claude.SUBMIT_MEMO_TOOL, broken)))
    with pytest.raises(claude.AgentError, match="invalid"):
        claude.write_memo(toolbox, client=client)


def test_memo_mode_requires_the_submit_tool(toolbox):
    client = FakeClient(response(text("Aqui va el memo en texto."), stop_reason="end_turn"))
    with pytest.raises(claude.AgentError, match=claude.SUBMIT_MEMO_TOOL):
        claude.write_memo(toolbox, client=client)


@pytest.mark.parametrize(
    ("stop_reason", "message"), [("refusal", "declined"), ("max_tokens", "truncated")]
)
def test_abnormal_stops_raise(toolbox, stop_reason, message):
    client = FakeClient(response(text(""), stop_reason=stop_reason))
    with pytest.raises(claude.AgentError, match=message):
        claude.ask(toolbox, "?", client=client)


def test_turn_limit(toolbox):
    loops = [response(tool_use("compare_decisions", {}, f"c{i}")) for i in range(3)]
    with pytest.raises(claude.AgentError, match="3 turns"):
        claude.ask(toolbox, "?", client=FakeClient(*loops), max_turns=3)


def test_missing_credentials_become_agent_error(toolbox):
    class NoCredentials(FakeClient):
        def _create(self, **kwargs):
            raise TypeError("Could not resolve authentication method.")

    with pytest.raises(claude.AgentError, match="ANTHROPIC_API_KEY"):
        claude.ask(toolbox, "?", client=NoCredentials())
