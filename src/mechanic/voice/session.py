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
from mechanic.voice.asr import Recognizer
from mechanic.voice.tts import Synthesiser
from mechanic.voice.vad import TurnDetector

# Said when a turn would otherwise end in silence.
NOTHING_TO_SAY = "Sorry, I did not catch that. Say it again?"

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
        self._cancel = asyncio.Event()
        self._turn: asyncio.Task | None = None

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
        if self._speaking and self.detector.is_speaking:
            await self._barge_in()
        for utterance in self.detector.push(pcm):
            if self._turn and not self._turn.done():
                continue  # already answering; ignore the overlap
            self._turn = asyncio.create_task(self.handle(utterance.audio))

    async def _barge_in(self) -> None:
        """The driver started talking over the answer: stop making noise and listen."""
        self._cancel.set()
        self._speaking = False
        await self._emit("flush", {"reason": "barge_in"})

    async def handle(self, speech: np.ndarray) -> TurnTimings:
        """Everything between the driver falling silent and the answer being spoken."""
        zero = time.monotonic()
        self._cancel.clear()
        timings = TurnTimings(speech_seconds=round(len(speech) / 16000, 2))

        transcript = await asyncio.to_thread(self.recognizer.transcribe, speech)
        timings.asr_ms = (time.monotonic() - zero) * 1000
        if not transcript.text:
            await self._emit("turn_skipped", {"reason": "nothing recognised"})
            return timings
        await self._emit("transcript", {"text": transcript.text, "ms": round(timings.asr_ms)})

        self._speaking = True
        try:
            async for sentence in self.agent.stream(transcript.text, on_event=self._agent_event):
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

        # A turn that produced nothing is the worst outcome for a voice agent: the driver is left
        # wondering whether the line dropped. Seen in testing when the model returned only its
        # internal reasoning and no spoken content.
        if timings.first_sentence_ms is None and not timings.barged_in:
            await self._emit("sentence", {"text": NOTHING_TO_SAY, "fallback": True})
            timings.first_sentence_ms = (time.monotonic() - zero) * 1000
            await self._speak(NOTHING_TO_SAY, zero, timings)

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
