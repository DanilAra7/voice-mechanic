"""Speech to text on the CPU, so the whole GPU stays available for the model and the voice.

Deliberately offline rather than streaming. Measured on day 4: the streaming build of this same
model needs 1.69 seconds of compute per second of speech and falls behind while the driver is
still talking, while one offline pass over a finished utterance costs about 100 ms — small
enough to disappear next to a model round.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mechanic.data.common import ROOT

SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = ROOT / "data" / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


@dataclass
class Transcript:
    text: str
    audio_seconds: float
    decode_ms: float


class Recognizer:
    """Wraps sherpa-onnx. The model loads on first use, not at import."""

    def __init__(self, model_dir: Path | None = None, threads: int = 4):
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.threads = threads
        self._rec = None

    def _load(self):
        if self._rec is None:
            import sherpa_onnx

            if not self.model_dir.exists():
                raise FileNotFoundError(f"ASR model missing at {self.model_dir}. See docs/NOTES.md for the download.")
            self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(self.model_dir / "encoder.int8.onnx"),
                decoder=str(self.model_dir / "decoder.int8.onnx"),
                joiner=str(self.model_dir / "joiner.int8.onnx"),
                tokens=str(self.model_dir / "tokens.txt"),
                num_threads=self.threads,
                provider="cpu",
                model_type="nemo_transducer",
            )
        return self._rec

    def warm_up(self) -> None:
        """Pay the model load and the first decode before a driver is waiting on them."""
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Transcript:
        import time

        if sample_rate != SAMPLE_RATE:
            audio = resample(audio, sample_rate, SAMPLE_RATE)
        rec = self._load()
        started = time.monotonic()
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        rec.decode_stream(stream)
        return Transcript(
            text=stream.result.text.strip(),
            audio_seconds=len(audio) / SAMPLE_RATE,
            decode_ms=(time.monotonic() - started) * 1000,
        )


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resampling: good enough for speech and free of an extra dependency."""
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    n = int(round(len(audio) * target_rate / source_rate))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1).astype(np.float32) if audio.ndim > 1 else audio.astype(np.float32)
