"""Refuse the drive before the model gets a say.

Day-3 benchmark: asked about petrol fumes in the cabin, none of the four candidate models led
with "stop the car" — every one of them started diagnosing instead. For symptoms that can hurt
someone, a confident helpful answer is worse than a slow one, so the warning is a rule here
rather than a behaviour we hope the model reproduces.

These patterns describe SYMPTOMS, not topics: "how do I check the brake fluid" is a how-to
question and must not trigger, while "the pedal goes to the floor" must.
"""

import re

PULL_OVER = "Stop driving and pull over as soon as it is safe."
DO_NOT_DRIVE = "Do not drive the car until this is checked."

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(smell|smells|smelling)\b.{0,20}\b(gas|gasoline|petrol|fuel)\b", re.I), PULL_OVER),
    (re.compile(r"\b(gas|gasoline|petrol|fuel)\b.{0,12}\bsmell", re.I), PULL_OVER),
    (re.compile(r"\bsmells?\s+like\s+(gas|gasoline|petrol|fuel|burning)\b", re.I), PULL_OVER),
    (re.compile(r"\bsmoke\b.{0,30}\b(hood|engine|bonnet|dash|vents?|car)\b", re.I), PULL_OVER),
    (re.compile(r"\b(on fire|flames|burning smell)\b", re.I), PULL_OVER),
    (re.compile(r"\bpedal\b.{0,30}\b(floor|all the way down)\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\b(no|lost|losing|failed?)\s+brakes?\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\bbrakes?\b.{0,20}\b(don'?t|do not|won'?t|not)\s+work", re.I), DO_NOT_DRIVE),
    (re.compile(r"\bbrake\s+failure\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\bsteering\b.{0,25}\b(locked|seized|stuck|failed)\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\blost\s+(my\s+)?steering\b", re.I), DO_NOT_DRIVE),
]


def safety_warning(user_text: str) -> str | None:
    """The sentence the driver must hear first, or None when nothing dangerous was described."""
    for pattern, line in _RULES:
        if pattern.search(user_text):
            return line
    return None
