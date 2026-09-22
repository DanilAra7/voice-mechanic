"""The spoken turn, with every model replaced by a stand-in.

What matters here is the order and the clock: the driver stops talking, text appears, a sentence
appears, sound starts — and an interruption stops all of it.
"""

import asyncio
import time

import numpy as np
import pytest
from helpers_voice import FakeRecognizer, FakeSynthesiser

from mechanic.voice.session import VoiceSession

SPEECH = np.zeros(16000, dtype=np.float32)


class FlushableDetector:
    """A detector that hands over one utterance the moment it is asked to flush."""

    def __init__(self, audio):
        self.audio = audio
        self.is_speaking = False

    def push(self, chunk):
        return iter(())

    def flush(self):
        from mechanic.voice.vad import Utterance

        yield Utterance(audio=self.audio, start_sample=0)

    def reset(self):
        pass


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
    session._speech_run_s = 0.0  # as `handle` sets it when a turn begins

    session.detector.is_speaking = True  # the tail of their own sentence still reads as speech
    await session.push_audio(np.zeros(1600, dtype=np.float32))  # 100 ms
    session.detector.is_speaking = False
    await session.push_audio(np.zeros(1600, dtype=np.float32))

    assert "flush" not in [e for e, _ in events], "interrupted itself on its own echo"


async def test_speaking_up_again_does_interrupt():
    session, events, _ = build()
    session.detector = QuietDetector()
    session._confirmed.set()  # they can hear the answer, so they can talk over it
    session._speaking = True
    session._speaking_since = time.monotonic() - 2.0  # the answer has been running a while
    session._speech_run_s = 0.0

    session.detector.is_speaking = True  # and they keep going, unlike an echo
    for _ in range(4):  # 400 ms, past BARGE_IN_SPEECH_S
        await session.push_audio(np.zeros(1600, dtype=np.float32))

    assert "flush" in [e for e, _ in events]


async def test_the_start_of_an_answer_cannot_be_interrupted():
    """Nobody interrupts a sentence that has not started; that reading is the previous question."""
    session, events, _ = build()
    session.detector = QuietDetector()
    session._confirmed.set()
    session._speaking = True
    session._speaking_since = time.monotonic()  # just began
    session.detector.is_speaking = True
    for _ in range(6):
        await session.push_audio(np.zeros(1600, dtype=np.float32))
    assert "flush" not in [e for e, _ in events]


async def test_a_click_is_not_a_question():
    from mechanic.voice.session import MIN_UTTERANCE_S
    from mechanic.voice.vad import Utterance

    session, _, _ = build()
    tiny = Utterance(audio=np.zeros(int(16000 * MIN_UTTERANCE_S / 2), dtype=np.float32), start_sample=0)
    session.detector = type("One", (), {"push": lambda self, c: iter([tiny]), "is_speaking": False})()
    await session.push_audio(np.zeros(1600, dtype=np.float32))
    assert session._turn is None


async def test_speech_during_the_grace_window_does_not_count_later():
    """The echo must not fill the interruption counter while it is being ignored."""
    session, events, _ = build()
    session.detector = QuietDetector()
    session._confirmed.set()
    session._speaking = True
    session._speaking_since = time.monotonic()
    session.detector.is_speaking = True

    for _ in range(6):  # 600 ms of echo, inside the grace window
        await session.push_audio(np.zeros(1600, dtype=np.float32))
    assert "flush" not in [e for e, _ in events]

    session._speaking_since = time.monotonic() - 5.0  # grace has now long expired
    session.detector.is_speaking = False  # and they are quiet
    await session.push_audio(np.zeros(1600, dtype=np.float32))
    session.detector.is_speaking = True  # one blip must not be enough
    await session.push_audio(np.zeros(1600, dtype=np.float32))
    assert "flush" not in [e for e, _ in events], "the grace-window echo still counted"


async def test_nothing_is_played_before_the_turn_is_confirmed():
    """The work runs early; only the sound waits. A half-question must never be heard."""
    from mechanic.voice import session as mod

    session, events, audio = build()
    original = mod.CONFIRM_EXTRA_S
    mod.CONFIRM_EXTRA_S = 5.0  # confirmation will not arrive during this test
    try:
        turn = asyncio.create_task(session.handle(SPEECH))
        await asyncio.sleep(0.4)  # long enough for recognition and the model
        assert [e for e, _ in events], "the turn did not start"
        assert "audio_start" not in [e for e, _ in events], "spoke before the turn was confirmed"
        assert audio == [], "sent audio before the turn was confirmed"
        session._confirmed.set()
        await asyncio.wait_for(turn, timeout=5)
    finally:
        mod.CONFIRM_EXTRA_S = original
    assert audio, "never spoke even after confirmation"


async def test_carrying_on_drops_the_answer_and_keeps_the_words():
    """A second utterance before confirmation means they had not finished, not that they cut in."""
    from mechanic.voice import session as mod
    from mechanic.voice.vad import Utterance

    session, events, audio = build()
    more = Utterance(audio=np.zeros(16000, dtype=np.float32), start_sample=0)
    pushes = [[], [more]]
    session.detector = type(
        "Continuing",
        (),
        {"push": lambda self, c: iter(pushes.pop(0) if pushes else []), "is_speaking": False},
    )()

    original = mod.CONFIRM_EXTRA_S
    mod.CONFIRM_EXTRA_S = 5.0  # confirmation will not arrive on its own
    try:
        await session.push_audio(np.zeros(1600, dtype=np.float32))  # nothing yet
        session._turn = asyncio.create_task(session.handle(SPEECH))
        await asyncio.sleep(0.3)
        session.recognizer.text = "what does that mean"
        await session.push_audio(np.zeros(1600, dtype=np.float32))  # they carry on
        follow_up = session._turn
        await asyncio.sleep(0.2)  # let the new turn clear the flag, then confirm it
        session._confirmed.set()
        await asyncio.wait_for(follow_up, timeout=5)
    finally:
        mod.CONFIRM_EXTRA_S = original

    assert "carry_on" in [e for e, _ in events]
    transcripts = [p["text"] for e, p in events if e == "transcript"]
    assert transcripts[-1] == "my idle feels rough what does that mean", "lost what they had said"


async def test_an_unheard_answer_cannot_be_interrupted():
    """Nothing has been played yet, so speech is them finishing the question, not cutting in."""
    session, events, _ = build()
    session.detector = QuietDetector()
    session._speaking = True
    session._speaking_since = time.monotonic() - 5.0  # grace long gone
    session.detector.is_speaking = True  # and they are talking
    for _ in range(6):
        await session.push_audio(np.zeros(1600, dtype=np.float32))
    assert "flush" not in [e for e, _ in events]


async def test_reset_stops_an_answer_that_is_already_playing():
    """ "New conversation" has to silence the agent: otherwise the previous answer keeps playing
    over the next question, and is then measured as part of it."""
    session, events, _ = build()
    session._turn = asyncio.create_task(session.handle(SPEECH))
    await asyncio.sleep(0.02)
    await session.reset()

    assert session._turn is None
    assert not session._speaking
    assert ("flush", {"reason": "reset"}) in events


async def test_the_driver_can_declare_the_turn_over():
    """Push-to-talk knows something the detector can only infer, and skips the silence wait."""
    session, events, audio = build()
    session.detector = FlushableDetector(SPEECH)
    await session.end_of_speech()
    await asyncio.gather(session._turn, return_exceptions=True)

    names = [e for e, _ in events]
    assert "transcript" in names and "audio_start" in names
    assert audio, "the answer has to be played without waiting for a confirmation"


async def test_a_held_button_keeps_the_answer_back():
    """A pause while the button is down is a pause, not the end of the question."""
    session, _, _ = build()
    session.start_of_speech()
    session._confirm_timer = asyncio.create_task(session._confirm_after(0.01))
    await asyncio.sleep(0.05)
    assert not session._confirmed.is_set()

    session.detector = FlushableDetector(SPEECH)
    await session.end_of_speech()
    assert session._confirmed.is_set()
    await asyncio.gather(session._turn, return_exceptions=True)


def test_latency_can_be_asked_for_before_a_turn_has_finished():
    """The browser reports what it waited the moment it hears the first sound, which is before
    the turn is over. Without this attribute existing from the start, the socket handler raised
    AttributeError and the connection dropped after the first exchange - a failure that looked
    like the network and was not."""
    session, _, _ = build()
    assert session.last_timings is None


class FillerThenSlowToolAgent:
    """Speaks before it looks, the way the real loop does when a tool will be slow."""

    async def stream(self, text, on_event=None, **kwargs):
        yield "Let me check that."
        if on_event:
            await on_event("tool_call", {"name": "search_forum", "duration_ms": 400})
        yield "Your fuel trim is high at idle."


async def test_the_stages_of_a_filler_turn_are_slices_not_milestones():
    """The turn a breakdown is worth having is the one subtraction gets wrong.

    When a tool will be slow the agent says a holding line first, so the first sentence — and the
    first sound — happen BEFORE the tool has run. Deriving the model's share by subtracting the
    tool time from the first sentence therefore goes negative on exactly these turns.
    """
    session, _, _ = build()
    session.agent = FillerThenSlowToolAgent()

    timings = await session.handle(SPEECH, confirmed=True)

    # The old, derived figure. Kept here so the reason for the change cannot quietly stop being true.
    assert timings.first_sentence_ms - timings.asr_ms - timings.tool_ms < 0

    assert timings.tool_ms == 400, "the whole turn's tool time is still reported"
    assert timings.tool_ms_to_audio == 0, "no tool had run when the driver heard the first word"
    assert timings.model_ms >= 0
    assert timings.hold_ms == 0.0, "the driver said they had finished; nothing to wait for"
    stages = timings.asr_ms + timings.model_ms + timings.tool_ms_to_audio + timings.tts_ms + timings.hold_ms
    assert abs(stages - timings.first_audio_ms) < 1e-6, "the slices must cover the wait exactly"


async def test_waiting_to_be_sure_the_driver_finished_is_charged_to_the_hold():
    """Deliberate silence is not the model being slow, and must not be reported as it."""
    session, _, _ = build(sentences=("Fuel trim is high at idle.",))

    timings = await session.handle(SPEECH)  # unconfirmed: the detector guessed from silence

    assert timings.hold_ms > 100, "the first sound is held back until the turn is certain"
    stages = timings.asr_ms + timings.model_ms + timings.tool_ms_to_audio + timings.tts_ms + timings.hold_ms
    assert abs(stages - timings.first_audio_ms) < 1e-6
