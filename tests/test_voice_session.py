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


def test_synthesiser_reports_a_rate_before_it_is_loaded():
    """The handshake sends the audio rate to the browser before anyone has spoken."""
    from mechanic.voice.tts import SAMPLE_RATE, Synthesiser

    assert Synthesiser().sample_rate == SAMPLE_RATE


async def test_a_turn_never_ends_in_silence():
    """A model that says nothing must not leave the driver wondering if the line dropped."""
    from mechanic.voice.session import NOTHING_TO_SAY

    session, events, audio = build(sentences=[])
    timings = await session.handle(SPEECH)

    assert session.synthesiser.said == [NOTHING_TO_SAY]
    assert timings.first_audio_ms is not None
    assert audio, "the fallback has to be spoken, not just logged"


async def test_late_audio_frames_are_not_dropped():
    """Frames handed over from the worker thread can land after it finishes."""
    session, _, audio = build(sentences=["One."])
    session.synthesiser.frames = 12
    await session.handle(SPEECH)
    assert len(audio) == 12


class QuietDetector:
    """Reports whatever `is_speaking` is set to and never produces an utterance."""

    is_speaking = False

    def push(self, _chunk):
        return iter(())


async def test_the_tail_of_the_drivers_own_words_is_not_an_interruption():
    """When an answer starts the driver has only just stopped; that is not them talking over it."""
    session, events, _ = build()
    session.detector = QuietDetector()
    session._speaking = True
    session._speech_run_s = 0.0           # as `handle` sets it when a turn begins

    session.detector.is_speaking = True   # the tail of their own sentence still reads as speech
    await session.push_audio(np.zeros(1600, dtype=np.float32))   # 100 ms
    session.detector.is_speaking = False
    await session.push_audio(np.zeros(1600, dtype=np.float32))

    assert "flush" not in [e for e, _ in events], "interrupted itself on its own echo"


async def test_speaking_up_again_does_interrupt():
    session, events, _ = build()
    session.detector = QuietDetector()
    session._speaking = True
    session._speech_run_s = 0.0

    session.detector.is_speaking = True   # and they keep going, unlike an echo
    for _ in range(4):                    # 400 ms, past BARGE_IN_SPEECH_S
        await session.push_audio(np.zeros(1600, dtype=np.float32))

    assert "flush" in [e for e, _ in events]


async def test_a_click_is_not_a_question():
    from mechanic.voice.session import MIN_UTTERANCE_S
    from mechanic.voice.vad import Utterance

    session, _, _ = build()
    tiny = Utterance(audio=np.zeros(int(16000 * MIN_UTTERANCE_S / 2), dtype=np.float32), start_sample=0)
    session.detector = type("One", (), {"push": lambda self, c: iter([tiny]), "is_speaking": False})()
    await session.push_audio(np.zeros(1600, dtype=np.float32))
    assert session._turn is None
