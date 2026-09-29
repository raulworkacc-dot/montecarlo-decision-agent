import json
import re

import pytest

from montecarlo_decisions.agent.tools import Toolbox, ToolInputError, summarize_result
from montecarlo_decisions.scenarios import DECISION_KEYS


@pytest.fixture(scope="module")
def toolbox(artifacts) -> Toolbox:
    return Toolbox(artifacts)


def test_definitions_are_strict_json_schemas(toolbox):
    for definition in toolbox.definitions():
        schema = definition["input_schema"]
        assert definition["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("drop_tables", {}, "Unknown tool"),
        ("get_lever_uplift", {"lever": "ads"}, "not one of"),
        ("get_decision_distribution", {}, "expects arguments"),
        ("compare_decisions", {"extra": 1}, "expects arguments"),
    ],
)
def test_invalid_calls_are_rejected(toolbox, name, arguments, message):
    with pytest.raises(ToolInputError, match=message):
        toolbox.call(name, arguments)


def _all_results(toolbox):
    yield "analyze_business", toolbox.call("analyze_business", {"channel": "all"})
    yield "get_ads_budget_regimes", toolbox.call("get_ads_budget_regimes", {})
    yield "compare_decisions", toolbox.call("compare_decisions", {})
    yield "get_validation_checks", toolbox.call("get_validation_checks", {})
    for lever in ("funnel", "webinar", "new_product"):
        yield "get_lever_uplift", toolbox.call("get_lever_uplift", {"lever": lever})
    for decision in DECISION_KEYS:
        yield (
            "get_decision_distribution",
            toolbox.call("get_decision_distribution", {"decision": decision}),
        )


def test_results_are_json_and_hide_the_ground_truth(toolbox, artifacts):
    hidden = [
        *artifacts.uplift["true_value_lift_per_opportunity_eur"],
        *artifacts.uplift["naive_value_lift_per_opportunity_eur"],
    ]
    patterns = [re.compile(rf"(?<![\d.]){re.escape(f'{value:.1f}')}(?!\d)") for value in hidden]
    for name, result in _all_results(toolbox):
        text = json.dumps(result, allow_nan=False)
        assert "true_" not in text, name
        assert "naive_" not in text, name
        assert "vs true" not in text, name
        leaked = [pattern.pattern for pattern in patterns if pattern.search(text)]
        assert not leaked, (name, leaked)
        assert summarize_result(name, result)


def test_validation_tool_keeps_outcomes(toolbox, artifacts):
    checks = toolbox.call("get_validation_checks", {})["checks"]
    assert [c["name"] for c in checks] == [c["name"] for c in artifacts.checks]
    assert [c["passed"] for c in checks] == [c["passed"] for c in artifacts.checks]


def test_distribution_matches_summary(toolbox, artifacts):
    result = toolbox.call("get_decision_distribution", {"decision": "funnel"})
    row = artifacts.summary.set_index("decision").loc["funnel"]
    assert result["expected_profit_eur"] == pytest.approx(row["expected_profit_eur"], abs=0.01)
    assert (
        result["assumptions"]["fixed_cost_eur"]
        == artifacts.config.decisions["funnel"].fixed_cost_eur
    )


def test_single_channel_filter(toolbox):
    result = toolbox.call("analyze_business", {"channel": "Email"})
    assert [row["channel"] for row in result["channels"]] == ["Email"]


def test_summaries_use_spanish_thousands(toolbox):
    text = summarize_result("compare_decisions", toolbox.call("compare_decisions", {}))
    assert "EUR" in text
    assert "," not in text.split("EUR")[0]
