import copy
import json

import pytest

from montecarlo_decisions.agent import memo as memo_module
from montecarlo_decisions.agent.tools import Toolbox


@pytest.fixture(scope="module")
def rules_memo(artifacts) -> dict:
    return memo_module.rule_based_memo(Toolbox(artifacts))


def test_rule_based_memo_follows_the_ranking(rules_memo, artifacts):
    ranking = artifacts.summary.sort_values("ranking")["decision"].tolist()
    assert rules_memo["source"] == "rules"
    assert rules_memo["model"] is None
    assert rules_memo["recommended_decision"] == ranking[0]
    assert rules_memo["runner_up"] == ranking[1]
    assert rules_memo["fingerprint"] == artifacts.fingerprint


def test_rule_based_memo_records_real_tool_calls(rules_memo):
    names = {step["name"] for step in rules_memo["tool_trace"]}
    assert {"compare_decisions", "get_decision_distribution", "get_validation_checks"} <= names


def test_rule_based_memo_is_valid(rules_memo):
    body = {key: rules_memo[key] for key in memo_module.MEMO_SCHEMA["required"]}
    assert memo_module.validate_memo_content(body) == body


def _valid_body(rules_memo) -> dict:
    return copy.deepcopy({key: rules_memo[key] for key in memo_module.MEMO_SCHEMA["required"]})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda body: body.pop("summary"), "missing"),
        (lambda body: body.update(recommended_decision="crypto"), "known decision"),
        (lambda body: body.update(runner_up=body["recommended_decision"]), "differ"),
        (lambda body: body.update(reasons=["  "]), "at least one"),
        (lambda body: body["audience_views"].pop("ceo"), "audience_views.ceo"),
    ],
)
def test_invalid_memos_are_rejected(rules_memo, mutate, message):
    body = _valid_body(rules_memo)
    mutate(body)
    with pytest.raises(ValueError, match=message):
        memo_module.validate_memo_content(body)


def test_lists_are_capped(rules_memo):
    body = _valid_body(rules_memo)
    body["findings"] = [f"item {i}" for i in range(10)]
    assert len(memo_module.validate_memo_content(body)["findings"]) == memo_module.MAX_ITEMS


def test_saved_memo_is_only_used_for_matching_results(tmp_path, rules_memo):
    path = tmp_path / "memo.json"
    path.write_text(json.dumps(rules_memo), encoding="utf-8")
    assert memo_module.load_memo(path, rules_memo["fingerprint"]) == rules_memo
    assert memo_module.load_memo(path, "stale") is None
    assert memo_module.load_memo(tmp_path / "missing.json", "x") is None
