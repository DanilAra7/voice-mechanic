"""Safety guardrail: symptoms must trigger, topics must not."""

import pytest

from mechanic.agent.safety import safety_warning

DANGEROUS = [
    "I smell gasoline inside the cabin while driving.",
    "There is a strong fuel smell when I stop.",
    "It smells like burning under the hood.",
    "My brake pedal goes almost to the floor. Can I drive to work tomorrow?",
    "I have no brakes at all.",
    "The brakes don't work properly anymore.",
    "There is smoke coming from under the hood.",
    "The steering locked up while I was turning.",
]

SAFE = [
    "How do I check the brake fluid level?",
    "Walk me through changing the oil and filter.",
    "My idle feels rough and the check engine light came on.",
    "How often should I replace the brake pads?",
    "What does code P0171 mean?",
    "The fuel economy got worse this month.",
    "How do I check the coolant level on this car?",
]


@pytest.mark.parametrize("text", DANGEROUS)
def test_dangerous_symptoms_warn(text):
    assert safety_warning(text) is not None, text


@pytest.mark.parametrize("text", SAFE)
def test_ordinary_questions_do_not_warn(text):
    assert safety_warning(text) is None, text
