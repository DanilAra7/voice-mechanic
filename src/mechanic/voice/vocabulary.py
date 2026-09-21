"""Put the words of the trade back into the transcript.

Parakeet is accurate on ordinary English - 3.3% word error on our recordings - and then loses
exactly the words this agent is for. Two seen on real speech, both changing what the driver
meant rather than how it reads:

    "fuel trims"          -> "field dreams"
    "the OBD adapter"     -> "the ad app server"

The proper cure is contextual biasing: hand the recogniser the jargon before it decodes. Kyutai
aside, sherpa-onnx needs a `bpe.model` for that, and `parakeet-tdt-0.6b-v2` does not ship one.
So this is a stopgap and should be read as one: a table of mishearings we have actually observed,
not a phonetic model. It cannot fix a mishearing nobody has met yet, and every entry earns its
place by having cost a real conversation.

Two rules keep it from doing harm. Phrases only - single common words are never rewritten, so
"server" on its own is left alone. And the replacement must be a domain term the tools can act
on, which is why the table is short and stays short.
"""

import re

# left: what the microphone produced. right: what the driver said.
HEARD_AS: dict[str, str] = {
    "field dreams": "fuel trims",
    "field dream": "fuel trim",
    "ad app server": "OBD adapter",
    "ad app adapter": "OBD adapter",
    "obd app server": "OBD adapter",
    "o b d adapter": "OBD adapter",
    "add up to her": "adapter",
    "cat converter": "catalytic converter",
    "mass air flow": "mass airflow",
    "p c v valve": "PCV valve",
    "check engine light": "check engine light",
}

_PATTERNS = [
    (re.compile(rf"\b{re.escape(heard)}\b", re.I), said) for heard, said in HEARD_AS.items() if heard != said.lower()
]


def repair(text: str) -> str:
    """Rewrite known mishearings. Returns the text unchanged when nothing matches."""
    for pattern, said in _PATTERNS:
        text = pattern.sub(said, text)
    return text
