"""The judging harnesses run on the rented GPU, where a typo costs an instance start.

Everything here is the part that does not need a model: parsing what the judge said, scoring a
ranking, and comparing verdicts against the answers a person read by hand.
"""

import pytest
import yaml

from mechanic.data.common import ROOT
from mechanic.evals.judge import VERDICTS, agreement, build_prompt, describe_car, parse_verdict
from mechanic.evals.retrieval_judge import ndcg, parse_scores, score_run


def test_the_judge_is_told_what_the_agent_could_not_see():
    """The whole point of the judge over keyword matching: it knows the injected fault."""
    text = describe_car({"vehicle": "audi_a4_b8", "fault": "vacuum_leak", "mode": "idle"})
    assert "vacuum_leak" in text and "Audi" in text
    assert "healthy" in describe_car({"vehicle": "honda_civic_10", "fault": None, "mode": "city"})


def test_prompt_carries_every_turn_of_the_conversation():
    scenario = {"setup": {"vehicle": "audi_a4_b8", "fault": "coolant_leak"}}
    result = {"turns": [{"user": "is it hot", "answer": "yes"}, {"user": "can I drive", "answer": "no"}]}
    prompt = build_prompt(scenario, result)
    assert "can I drive" in prompt and "coolant_leak" in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"verdict": "wrong", "why": "drains through the filler cap"}', "wrong"),
        ('here you go:\n{"verdict":"weak","why":"invented a component"}', "weak"),
        ("I would call this correct.", "correct"),
        ("no idea what to say", "unparsed"),
    ],
)
def test_parse_verdict(raw, expected):
    verdict, _ = parse_verdict(raw)
    assert verdict == expected


def test_agreement_separates_lenient_from_harsh():
    """A judge that waves everything through and one that fails everything both score badly,
    and the report has to say which way it went."""
    judged = [
        {"id": "a", "verdict": "correct"},  # hand said wrong - too lenient
        {"id": "b", "verdict": "wrong"},  # hand said correct - too harsh
        {"id": "c", "verdict": "correct"},
        {"id": "d", "verdict": "weak"},  # hand said correct: disagrees, but same side
    ]
    gold = {"a": "wrong", "b": "correct", "c": "correct", "d": "correct"}
    out = agreement(judged, gold)
    assert out["compared"] == 4
    assert out["exact_agreement"] == 25.0
    assert out["correct_vs_not_agreement"] == 50.0
    assert out["judge_too_lenient"] == 1 and out["judge_too_harsh"] == 1


def test_hand_read_verdicts_are_wellformed():
    """This file is the yardstick the judge is measured against; a typo in it silently moves
    the yardstick."""
    gold = yaml.safe_load((ROOT / "evals" / "content_gold.yaml").read_text())
    ids = {g["id"] for g in gold}
    assert len(ids) == len(gold), "duplicate id in the gold set"
    scenario_ids = {s["id"] for s in yaml.safe_load((ROOT / "evals" / "scenarios.yaml").read_text())}
    assert ids <= scenario_ids, ids - scenario_ids
    for entry in gold:
        assert entry["verdict"] in VERDICTS, entry
        if entry["verdict"] != "correct":
            assert entry.get("why"), f"{entry['id']} is marked {entry['verdict']} with no reason"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"scores": [2, 0, 1]}', [2, 0, 1]),
        ("2 0 1 1", [2, 0, 1, 1]),
        ('{"scores": [2, 0]}', None),  # wrong length is not a judgement, it is a guess
        ("nothing here", None),
    ],
)
def test_parse_scores(raw, expected):
    assert parse_scores(raw, len(expected) if expected else 3) == expected


def test_ndcg_rewards_putting_the_good_passage_first():
    assert ndcg([2, 1, 0, 0], 10) == 1.0
    assert ndcg([0, 0, 1, 2], 10) < ndcg([1, 2, 0, 0], 10) < 1.0
    assert ndcg([0, 0, 0], 10) == 0.0


def test_score_run_counts_only_answers_as_useful():
    """Grade 1 is context, not an answer; precision must not quietly count it."""
    out = score_run([[2, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    assert out["queries"] == 2
    assert out["precision@5"] == 10.0  # one grade-2 passage across ten slots
    assert out["useful_in_top5_pct"] == 50.0
