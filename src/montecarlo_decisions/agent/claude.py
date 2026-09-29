"""Claude tool-use agent over the analytical toolbox.

A manual agentic loop (rather than the SDK's beta tool runner) keeps every step
explicit and testable with a fake client: the loop is bounded by ``max_turns``, every
tool input is validated before running, tool errors go back to the model as
``is_error`` results, and refusals or truncated turns stop the run with a clear error.

Two entry points:

* :func:`ask`: free-form question -> answer text + tool trace.
* :func:`write_memo`: the agent must finish by calling ``submit_decision_memo`` with a
  schema-valid memo, which the dashboard renders.

Requests use server-side model fallbacks (``fallbacks="default"``): if the primary
model declines a request, the API retries it on a fallback model within the same call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from montecarlo_decisions.agent.memo import MEMO_SCHEMA, finalize_memo, trace_entry
from montecarlo_decisions.agent.tools import Toolbox, ToolInputError, summarize_result

DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 12
MAX_TOKENS = 16_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
SUBMIT_MEMO_TOOL = "submit_decision_memo"

SYSTEM_PROMPT = """You are a senior growth-strategy analyst advising the leadership of a \
digital-marketing business. Four initiatives compete for budget over the next planning \
horizon: improving the funnel (landing, CTA, lead magnet, checkout), a sales webinar, \
doubling paid-ads spend, and launching a new higher-ticket product.

You have read-only tools over a completed analysis: a historical baseline, counterfactual \
uplift estimated by ML models (with bootstrap intervals), paid-media saturation evidence, \
Monte Carlo distributions of incremental profit per decision and the quality checks of \
the run. Ground every claim in tool results and quote the numbers you rely on. Treat \
estimates as estimates: mention uncertainty where it changes the conclusion, and say so \
when the data cannot answer a question. Write in clear executive Spanish, without \
markdown headings."""

MEMO_INSTRUCTIONS = f"""Prepare the decision memo for the dashboard. Before writing, \
check the validation results, compare all decisions, inspect the distribution of each \
decision, and review the uplift of the funnel, webinar and new-product levers and the \
paid-media budget regimes. Each list item must add something a reader could not get \
from the ranking alone (a risk, a threshold, a trigger to change course). Finish by \
calling {SUBMIT_MEMO_TOOL} exactly once with the memo; do not write the memo as text."""


class AgentError(RuntimeError):
    """The agent could not complete the task (refusal, truncation, turn limit...)."""


@dataclass
class AgentRun:
    model: str
    answer: str = ""
    memo: dict[str, Any] | None = None
    trace: list[dict[str, str]] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def submit_memo_tool() -> dict[str, Any]:
    return {
        "name": SUBMIT_MEMO_TOOL,
        "description": "Submit the final decision memo. Call once, after the analysis.",
        "input_schema": MEMO_SCHEMA,
        "strict": True,
    }


def _text(content: list[Any]) -> str:
    return "\n".join(block.text for block in content if block.type == "text").strip()


def _run_tool(toolbox: Toolbox, block: Any, run: AgentRun) -> dict[str, Any]:
    arguments = block.input if isinstance(block.input, dict) else {}
    try:
        result = toolbox.call(block.name, arguments)
    except (ToolInputError, KeyError) as error:
        run.trace.append(trace_entry(block.name, arguments, f"Error: {error}"))
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"Error: {error}",
            "is_error": True,
        }
    run.trace.append(trace_entry(block.name, arguments, summarize_result(block.name, result)))
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def run_agent(
    toolbox: Toolbox,
    prompt: str,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TURNS,
    want_memo: bool = False,
) -> AgentRun:
    """Drive the tool-use loop until the model answers (or submits the memo)."""
    client = client or anthropic.Anthropic()
    tools = toolbox.definitions() + ([submit_memo_tool()] if want_memo else [])
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    run = AgentRun(model=model)

    while run.turns < max_turns:
        run.turns += 1
        try:
            response = client.beta.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
                thinking={"type": "adaptive"},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except TypeError as error:  # raised by the SDK when no credentials can be resolved
            raise AgentError(
                f"Claude API unavailable: {error}. Set ANTHROPIC_API_KEY (see .env.example)."
            ) from error
        run.model = response.model
        run.input_tokens += response.usage.input_tokens
        run.output_tokens += response.usage.output_tokens

        if response.stop_reason == "refusal":
            raise AgentError("The model declined the request.")
        if response.stop_reason == "max_tokens":
            raise AgentError("The model response was truncated (max_tokens).")

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            run.answer = _text(response.content)
            if want_memo:
                raise AgentError(f"The agent finished without calling {SUBMIT_MEMO_TOOL}.")
            return run

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            if block.name == SUBMIT_MEMO_TOOL and want_memo:
                run.memo = dict(block.input)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": "Memo received."}
                )
            else:
                results.append(_run_tool(toolbox, block, run))
        if run.memo is not None:
            run.answer = _text(response.content)
            return run
        messages.append({"role": "user", "content": results})

    raise AgentError(f"No final answer after {max_turns} turns.")


def ask(toolbox: Toolbox, question: str, **kwargs: Any) -> AgentRun:
    return run_agent(toolbox, question, **kwargs)


def write_memo(toolbox: Toolbox, **kwargs: Any) -> dict[str, Any]:
    """Ask the agent for the memo and return it validated, with its tool trace."""
    run = run_agent(toolbox, MEMO_INSTRUCTIONS, want_memo=True, **kwargs)
    try:
        return finalize_memo(
            run.memo or {},
            source="claude",
            model=run.model,
            fingerprint=toolbox.artifacts.fingerprint,
            tool_trace=run.trace,
        )
    except ValueError as error:
        raise AgentError(f"The submitted memo is invalid: {error}") from error
