"""Deciding that the driver has finished talking.

This is where the latency clock starts: every millisecond of silence we wait before answering
is added directly to the reply, and it is the one delay no amount of model tuning can recover.
Wait too little and the agent talks over someone mid-sentence; wait too long and it feels like
a radio. `min_silence_s` is that dial, and it is deliberately easy to find.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mechanic.data.common import ROOT

SAMPLE_RATE = 16000
DEFAULT_MODEL = ROOT / "data" / "models" / "silero_vad_v5.onnx"


@dataclass
class Utterance:
    """One stretch of speech, with where it sat in the incoming stream."""

    audio: np.ndarray
    start_sample: int
    # Audio time between this utterance's last word and the newest sample the detector had been
    # given when it finally admitted the turn was over. In hands-free this is the single most
    # expensive stage of a turn and it happens before any of our clocks start, so without it the
    # server measures a wait that begins half a second after the driver stopped talking. Zero
    # when a button declared the turn over: there was nothing to wait for.
    trailing_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class TurnDetector:
    def __init__(
        self,
        model_path: Path | None = None,
        threshold: float = 0.5,
        min_silence_s: float = 0.35,
        min_speech_s: float = 0.25,
        buffer_s: float = 30.0,
    ):
        self.model_path = Path(model_path or DEFAULT_MODEL)
        self.threshold = threshold
        self.min_silence_s = min_silence_s
        self.min_speech_s = min_speech_s
        self.buffer_s = buffer_s
        self._vad = None
        # Samples handed to the detector so far, so an utterance can say how long it sat waiting
        # for its own silence to be long enough.
        self._samples = 0

    def _load(self):
        if self._vad is None:
            import sherpa_onnx

            if not self.model_path.exists():
                raise FileNotFoundError(f"VAD model missing at {self.model_path}. See docs/NOTES.md for the download.")
            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(self.model_path)
            config.silero_vad.threshold = self.threshold
            config.silero_vad.min_silence_duration = self.min_silence_s
            config.silero_vad.min_speech_duration = self.min_speech_s
            config.sample_rate = SAMPLE_RATE
            self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=self.buffer_s)
        return self._vad

    @property
    def is_speaking(self) -> bool:
        """True while the driver is mid-utterance — the signal to stop talking over them."""
        return bool(self._load().is_speech_detected())

    def push(self, chunk: np.ndarray) -> Iterator[Utterance]:
        """Feed microphone audio; yields each utterance once its trailing silence is long enough."""
        vad = self._load()
        chunk = np.asarray(chunk, dtype=np.float32)
        vad.accept_waveform(chunk)
        self._samples += len(chunk)
        while not vad.empty():
            segment = vad.front
            samples = np.asarray(segment.samples, dtype=np.float32)
            waited = max(0, self._samples - (segment.start + len(samples)))
            yield Utterance(audio=samples, start_sample=segment.start, trailing_s=waited / SAMPLE_RATE)
            vad.pop()

    def flush(self) -> Iterator[Utterance]:
        """Close whatever is being said right now instead of waiting for the silence to prove it.

        Push-to-talk hands us something the detector can only ever infer: the moment the driver
        says they are done. Taking their word for it removes the whole silence wait, which is the
        single most expensive stage of a turn.
        """
        vad = self._load()
        vad.flush()
        while not vad.empty():
            segment = vad.front
            # trailing_s stays 0: the driver said they had finished, so nothing was waited out.
            yield Utterance(audio=np.asarray(segment.samples, dtype=np.float32), start_sample=segment.start)
            vad.pop()

    def reset(self) -> None:
        self._samples = 0
        self._load().reset()
