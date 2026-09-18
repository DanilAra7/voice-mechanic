"""One spoken conversation: microphone in, speech out, with the clock running.

Transport-agnostic on purpose. The WebSocket endpoint only moves bytes; everything about when
the driver stopped talking, what was said, which tools ran and when sound came back lives here,
so it can be tested without a browser.

The timings are a deliverable, not debug output — the site has to let someone measure the
latency themselves — so every stage is stamped against one zero point: the moment the turn
detector decided the driver had finished.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from mechanic.agent.loop import AgentLoop
from mechanic.voice.asr import SAMPLE_RATE, Recognizer
from mechanic.voice.tts import Synthesiser
from mechanic.voice.vad import TurnDetector

# Said when a turn would otherwise end in silence.
NOTHING_TO_SAY = "Sorry, I did not catch that. Say it again?"
# Shorter than this is a cough, a door, a click — answering it wastes a turn and confuses the driver.
MIN_UTTERANCE_S = 0.4
# Someone genuinely talking over the answer keeps going; a blip is buffered audio or a noise.
# Requiring the speech to last stops the agent cutting itself off on the tail of the last question.
BARGE_IN_SPEECH_S = 0.25
# The detector's buffer still holds the end of the question when the answer begins. Nobody can
# genuinely interrupt an answer before it has been going this long, so ignore anything sooner.
BARGE_IN_GRACE_S = 0.8
# The detector calls a turn over after a short silence, which is fast but splits ordinary speech:
# "I am getting a code P0171, what does that mean?" holds a pause of well over half a second.
# So the turn is treated as provisional for this much longer, and nothing is played before then —
# the work happens early, only the speaking waits. If the driver turns out to be mid-sentence they
# never hear a thing, and their words so far are carried into the next turn. This is on top of the
# silence the detector already waited for, putting the total at about the 0.9 s that measurement
# showed an ordinary spoken question can contain. A timer, not a count of incoming audio: a client
# that stops sending must not leave the agent mute forever.
CONFIRM_EXTRA_S = 0.55

Event = Callable[[str, dict], Awaitable[None]]
Audio = Callable[[np.ndarray, int], Awaitable[None]]


@dataclass
class TurnTimings:
    """Milliseconds from the driver falling silent. This is what the latency panel shows."""

    asr_ms: float | None = None
    first_sentence_ms: float | None = None
    first_audio_ms: float | None = None
    total_ms: float | None = None
    speech_seconds: float | None = None
    tools: list[str] = field(default_factory=list)
    barged_in: bool = False
    # The driver was still mid-question: this turn was dropped before anything was played.
    carried_on: bool = False


class VoiceSession:
    def __init__(
        self,
        agent: AgentLoop,
        recognizer: Recognizer | None = None,
        synthesiser: Synthesiser | None = None,
        detector: TurnDetector | None = None,
        on_event: Event | None = None,
        on_audio: Audio | None = None,
    ):
        self.agent = agent
        self.recognizer = recognizer or Recognizer()
        self.synthesiser = synthesiser or Synthesiser()
        self.detector = detector or TurnDetector()
        self.on_event = on_event
        self.on_audio = on_audio
        self._speaking = False
        self._speaking_since = 0.0
        # How long the detector has been hearing speech while we are answering. Barge-in needs
        # this to last: when an answer starts the driver has only just stopped, and the tail of
        # their own question still reads as speech, so any instant trigger fires on the echo.
        self._speech_run_s = 0.0
        self._cancel = asyncio.Event()
        self._turn: asyncio.Task | None = None
        self._confirmed = asyncio.Event()
        self._confirm_timer: asyncio.Task | None = None
        self._continued = False
        # What a provisional turn had heard before the driver carried on; prepended to what follows.
        self._pending_prefix = ""

    async def _emit(self, event: str, payload: dict) -> None:
        if self.on_event:
            await self.on_event(event, payload)

    async def warm_up(self) -> None:
        """Load every model before a driver is on the line; the first use of each costs seconds."""
        await asyncio.to_thread(self.recognizer.warm_up)
        await asyncio.to_thread(self.synthesiser.warm_up)
        await self._emit("ready", {})

    async def push_audio(self, pcm: np.ndarray) -> None:
        """Feed one chunk of microphone audio. Utterances are handled as they complete."""
        chunk_s = len(pcm) / SAMPLE_RATE
        answering_for = time.monotonic() - self._speaking_since if self._speaking else 0.0
        in_grace = self._speaking and answering_for <= BARGE_IN_GRACE_S
        if self.detector.is_speaking and not in_grace:
            self._speech_run_s += chunk_s
        else:
            # Nothing heard during the grace window counts, including towards the run: it is the
            # end of the question they just asked, still working through the detector's buffer.
            self._speech_run_s = 0.0

        # Only a confirmed turn can be interrupted: before that nothing has been played, so there
        # is nothing to talk over — speech then means they are still finishing their question.
        if self._confirmed.is_set() and self._speaking and not in_grace and self._speech_run_s >= BARGE_IN_SPEECH_S:
            await self._barge_in()
        for utterance in self.detector.push(pcm):
            if utterance.duration_s < MIN_UTTERANCE_S:
                continue  # a cough or a click, not a question
            if self._turn is not None and not self._turn.done():
                if self._confirmed.is_set():
                    continue  # cutting into an answer they can hear; barge-in deals with that
                # A whole second utterance while the first is still unconfirmed means they had
                # not finished. Far safer than reacting to "speech is audible", which the tail of
                # their own question sets off just as reliably.
                await self._carry_on()
                await asyncio.gather(self._turn, return_exceptions=True)
            self._turn = asyncio.create_task(self.handle(utterance.audio))

    async def _confirm_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self._confirmed.set()

    async def _carry_on(self) -> None:
        """Not an interruption: the driver had not finished. Drop the answer before it is heard."""
        self._continued = True
        if self._confirm_timer:
            self._confirm_timer.cancel()
        self._cancel.set()
        self._speaking = False
        self._speech_run_s = 0.0
        await self._emit("carry_on", {"heard_so_far": self._pending_prefix})

    async def _barge_in(self) -> None:
        """The driver started talking over the answer: stop making noise and listen."""
        self._cancel.set()
        self._speaking = False
        self._speech_run_s = 0.0
        await self._emit("flush", {"reason": "barge_in"})

    async def handle(self, speech: np.ndarray) -> TurnTimings:
        """Everything between the driver falling silent and the answer being spoken."""
        zero = time.monotonic()
        self._cancel.clear()
        self._confirmed.clear()
        self._continued = False
        self._speech_run_s = 0.0  # they have only just stopped; do not read the tail as new speech
        self._confirm_timer = asyncio.create_task(self._confirm_after(CONFIRM_EXTRA_S))
        timings = TurnTimings(speech_seconds=round(len(speech) / 16000, 2))

        transcript = await asyncio.to_thread(self.recognizer.transcribe, speech)
        timings.asr_ms = (time.monotonic() - zero) * 1000
        said = f"{self._pending_prefix} {transcript.text}".strip()
        if not said:
            await self._emit("turn_skipped", {"reason": "nothing recognised"})
            return timings
        self._pending_prefix = said   # kept in case this turn turns out to be half a question
        await self._emit("transcript", {"text": said, "ms": round(timings.asr_ms)})

        self._speaking = True
        self._speaking_since = time.monotonic()
        try:
            async for sentence in self.agent.stream(said, on_event=self._agent_event):
                if self._cancel.is_set():
                    timings.barged_in = True
                    break
                now = (time.monotonic() - zero) * 1000
                if timings.first_sentence_ms is None:
                    timings.first_sentence_ms = now
                await self._emit("sentence", {"text": sentence, "ms": round(now)})
                await self._speak(sentence, zero, timings)
        finally:
            self._speaking = False

        if self._continued:
            if self._confirm_timer:
                self._confirm_timer.cancel()
            timings.carried_on = True
            await self._emit("turn_end", {k: v for k, v in asdict(timings).items() if v is not None})
            return timings

        self._pending_prefix = ""     # the answer was heard, so nothing is left hanging

        # A turn that produced nothing is the worst outcome for a voice agent: the driver is left
        # wondering whether the line dropped. Seen in testing when the model returned only its
        # internal reasoning and no spoken content.
        if timings.first_sentence_ms is None and not timings.barged_in:
            await self._emit("sentence", {"text": NOTHING_TO_SAY, "fallback": True})
            timings.first_sentence_ms = (time.monotonic() - zero) * 1000
            await self._speak(NOTHING_TO_SAY, zero, timings)

        # Only now: the fallback above still has to wait for the same confirmation as any answer.
        if self._confirm_timer:
            self._confirm_timer.cancel()
        timings.total_ms = (time.monotonic() - zero) * 1000
        await self._emit("turn_end", {k: v for k, v in asdict(timings).items() if v is not None})
        return timings

    async def _agent_event(self, event: str, payload: dict) -> None:
        if event == "tool_call":
            await self._emit("tool", payload)

    async def _speak(self, sentence: str, zero: float, timings: TurnTimings) -> None:
        """Synthesise and ship audio frames as they appear, so sound starts before the sentence ends."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_frame(chunk: np.ndarray) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, chunk)

        work = asyncio.create_task(asyncio.to_thread(self.synthesiser.say, sentence, on_frame))
        sample_rate = self.synthesiser_sample_rate

        async def drain_one() -> bool:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.05)
            except TimeoutError:
                return True
            if timings.first_audio_ms is None and not self._confirmed.is_set():
                # Hold the very first sound until the turn is certain. Everything up to here —
                # recognition, the model, the synthesiser — has already run.
                while not self._confirmed.is_set() and not self._cancel.is_set():
                    await asyncio.sleep(0.02)
            if self._cancel.is_set():
                timings.barged_in = True
                return False
            if timings.first_audio_ms is None:
                timings.first_audio_ms = (time.monotonic() - zero) * 1000
                await self._emit("audio_start", {"ms": round(timings.first_audio_ms)})
            if self.on_audio:
                await self.on_audio(chunk, sample_rate)
            return True

        while not work.done() or not queue.empty():
            if not await drain_one():
                break
        await work
        # The worker thread hands frames over through call_soon_threadsafe, so the last of them can
        # still be sitting in the loop's callback queue when the thread is already finished.
        # Without this the final frames of a sentence are simply dropped.
        await asyncio.sleep(0)
        while not queue.empty() and not self._cancel.is_set():
            await drain_one()

    @property
    def synthesiser_sample_rate(self) -> int:
        return getattr(self.synthesiser, "sample_rate", 24000)
