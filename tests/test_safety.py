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


def test_the_rules_survive_what_a_recogniser_actually_writes():
    """Measured on real speech: "I smelled gasoline inside the cabin" went straight through,
    because the rule knew smell, smells and smelling but not the past tense. The agent then
    talked about fuel trims to someone sitting in petrol fumes."""
    for said in [
        "I smelled gasoline inside the cabin while driving.",
        "It smells of petrol in here.",
        "There is smoke coming out from under the hood.",
        "The steering just went really heavy on me.",
        "Coming down a long hill the brakes went soft and I nearly didn't stop.",
    ]:
        assert safety_warning(said) is not None, said


def test_the_pump_is_not_an_emergency():
    """Refuelling is the one place a driver smells petrol and nothing is wrong. A warning that
    fires there is one they learn to talk over."""
    assert safety_warning("I smelled gasoline while I was filling up at the petrol station.") is None
    assert safety_warning("Is it normal to smell fuel at the pump?") is None


def test_fumes_where_the_driver_sits_are_still_an_emergency():
    """The refuelling exception must not swallow the dangerous case that mentions the pump."""
    assert safety_warning("I smell fuel in the cabin after filling up.") is not None
    assert safety_warning("Since the petrol station there is a smell of fuel through the vents.") is not None


def test_burning_is_caught_however_it_is_phrased():
    """Measured: "It smells like something is burning and I can feel the heat" produced a calm
    report that the coolant was fine. The word that matters sits three words after the one that
    introduces it, which is how people say it."""
    for said in [
        "It smells like something is burning and I can feel the heat.",
        "There is a burning smell in the car.",
        "Something is burning under the bonnet.",
        "It smells of burning.",
    ]:
        assert safety_warning(said) is not None, said
