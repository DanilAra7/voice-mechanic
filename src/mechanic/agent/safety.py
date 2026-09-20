"""Refuse the drive before the model gets a say.

Day-3 benchmark: asked about petrol fumes in the cabin, none of the four candidate models led
with "stop the car" — every one of them started diagnosing instead. For symptoms that can hurt
someone, a confident helpful answer is worse than a slow one, so the warning is a rule here
rather than a behaviour we hope the model reproduces.

These patterns describe SYMPTOMS, not topics: "how do I check the brake fluid" is a how-to
question and must not trigger, while "the pedal goes to the floor" must.

They also have to survive a recogniser. Measured on real speech 2026-09-20: a driver said "I
smelled gasoline inside the cabin while driving", the transcript came back in the past tense,
and the rule — which knew "smell", "smells" and "smelling" — let it through. The agent then
discussed fuel trims with someone sitting in petrol fumes. Every verb here now carries its
endings, because the words that reach this function are whatever the microphone made of them.
"""

import re

PULL_OVER = "Stop driving and pull over as soon as it is safe."
DO_NOT_DRIVE = "Do not drive the car until this is checked."

SMELL = r"smell(?:s|ed|ing)?"
FUEL = r"gas|gasoline|petrol|fuel"

# Refuelling is the one place a driver smells petrol and nothing is wrong. Without this the rule
# shouts at "I smelled petrol while filling up", and a warning that cries wolf gets talked over.
_AT_THE_PUMP = re.compile(r"\b(fill(ing|ed)? up|at the pump|petrol station|gas station|refuel)", re.I)
# Except that fumes where the driver is sitting are dangerous wherever they started.
_IN_THE_CAR = re.compile(r"\b(cabin|inside the car|in the car|interior|vents?)\b", re.I)

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{SMELL}\b.{{0,20}}\b({FUEL})\b", re.I), PULL_OVER),
    (re.compile(rf"\b({FUEL})\b.{{0,12}}\b{SMELL}", re.I), PULL_OVER),
    # "smells like something is burning" — the word that matters is several words away from the
    # one that introduces it, which is how people actually say this.
    (re.compile(rf"\b{SMELL}\s+(of|like)?\s*(\w+\s+){{0,3}}({FUEL}|burning)\b", re.I), PULL_OVER),
    (re.compile(r"\bsomething\s+is\s+burning\b", re.I), PULL_OVER),
    (re.compile(r"\bsmok(e|ing|y)\b.{0,30}\b(hood|engine|bonnet|dash|vents?|car)\b", re.I), PULL_OVER),
    (re.compile(r"\b(on fire|flames|burning smell|smell of burning)\b", re.I), PULL_OVER),
    (re.compile(r"\bpedal\b.{0,30}\b(floor|all the way down)\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\b(no|lost|losing|loses|failed?|failing)\s+brakes?\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\bbrakes?\b.{0,20}\b(don'?t|do not|won'?t|did ?n'?t|not)\s+work", re.I), DO_NOT_DRIVE),
    (re.compile(r"\bbrakes?\b.{0,20}\b(went|go|going|gone|feel|felt)\s+(soft|spongy)\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\bbrake\s+fail(ure|ed|ing)?\b", re.I), DO_NOT_DRIVE),
    (re.compile(r"\bsteering\b.{0,25}\b(lock(ed|s)?|seiz(ed|es)?|stuck|fail(ed|s|ing)?)\b", re.I), DO_NOT_DRIVE),
    (
        re.compile(r"\bsteering\b.{0,25}\b(went|going|gone|got|is)\s+(really\s+)?(heavy|stiff|hard)\b", re.I),
        DO_NOT_DRIVE,
    ),
    (re.compile(r"\b(lost|losing|loses)\s+(my\s+)?steering\b", re.I), DO_NOT_DRIVE),
]


def safety_warning(user_text: str) -> str | None:
    """The sentence the driver must hear first, or None when nothing dangerous was described."""
    at_the_pump = bool(_AT_THE_PUMP.search(user_text)) and not _IN_THE_CAR.search(user_text)
    for pattern, line in _RULES:
        if pattern.search(user_text):
            if line is PULL_OVER and at_the_pump:
                continue
            return line
    return None
