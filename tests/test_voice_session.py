"""The spoken turn, with every model replaced by a stand-in.

What matters here is the order and the clock: the driver stops talking, text appears, a sentence
appears, sound starts — and an interruption stops all of it.
"""

import asyncio

import numpy as np
import pytest
from helpers_voice import FakeRecognizer, FakeSynthesiser

from mechanic.voice.session import VoiceSession

SPEECH = np.zeros(16000, dtype=np.float32)


class FakeAgent:
    def __init__(self, sentences):
        self.sentences = sentences
        self.asked = []

    async def stream(self, text, on_event=None, **kwargs):
        self.asked.append(text)
        for s in self.sentences:
            if on_event:
                await on_event("tool_call", {"name": "read_live_data", "duration_ms": 1})
            yield s


def build(sentences=("Checking your live data.", "Fuel trim is high at idle."), text="my idle feels rough"):
    events, audio = [], []

    async def on_event(event, payload):
        events.append((event, payload))

    async def on_audio(chunk, rate):
        audio.append((len(chunk), rate))

    session = VoiceSession(
        agent=FakeAgent(list(sentences)),
        recognizer=FakeRecognizer(text),
        synthesiser=FakeSynthesiser(),
        on_event=on_event,
        on_audio=on_audio,
    )
    return session, events, audio


async def test_a_turn_reports_each_stage_in_order():
    session, events, audio = build()
    timings = await session.handle(SPEECH)

    names = [e for e, _ in events]
    assert names.index("transcript") < names.index("sentence") < names.index("audio_start")
    assert "turn_end" in names
    assert timings.asr_ms <= timings.first_sentence_ms <= timings.first_audio_ms <= timings.total_ms
    assert audio, "the driver has to actually get audio back"


async def test_every_sentence_is_spoken():
    session, _, _ = build()
    await session.handle(SPEECH)
    assert session.synthesiser.said == ["Checking your live data.", "Fuel trim is high at idle."]


async def test_silence_that_recognises_as_nothing_is_dropped():
    session, events, audio = build(text="")
    timings = await session.handle(SPEECH)
    assert [e for e, _ in events] == ["turn_skipped"]
    assert timings.first_sentence_ms is None
    assert audio == []


async def test_barge_in_stops_the_answer():
    session, events, _ = build(sentences=["One.", "Two.", "Three."])
    original = session.synthesiser.say

    def interrupt(text, on_frame=None):
        if text == "Two.":
            session._cancel.set()
        return original(text, on_frame)

    session.synthesiser.say = interrupt
    timings = await session.handle(SPEECH)

    assert timings.barged_in is True
    assert "Three." not in session.synthesiser.said, "kept talking after being interrupted"


async def test_timings_are_measured_from_the_end_of_speech():
    session, _, _ = build()

    slow = session.recognizer.transcribe

    def delayed(audio, sample_rate=16000):
        import time as t

        t.sleep(0.05)
        return slow(audio, sample_rate)

    session.recognizer.transcribe = delayed
    timings = await session.handle(SPEECH)
    assert timings.asr_ms >= 50, "the clock must include recognition, not start after it"


@pytest.mark.parametrize("chunks", [1, 4])
async def test_pushing_audio_does_not_block(chunks):
    """Microphone frames must keep flowing while a turn is being answered."""
    session, _, _ = build()
    session.detector = type("NoTurns", (), {"push": lambda self, c: iter(()), "is_speaking": False})()
    await asyncio.wait_for(
        asyncio.gather(*(session.push_audio(np.zeros(1600, dtype=np.float32)) for _ in range(chunks))),
        timeout=1.0,
    )
