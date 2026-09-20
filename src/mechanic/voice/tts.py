"""Turning the agent's sentences into sound, a frame at a time.

Kyutai was chosen on day 3 for one reason: it emits audio frames while it is still working, so
the driver hears the first word about 350 ms in rather than waiting out the whole sentence — a
non-streaming synthesiser of the same quality made them wait 1186 ms.

The model lives on the GPU beside the language model, which leaves very little room: measured
together they peak at 15 875 MiB of the card's 16 376. Nothing else may be loaded alongside.
"""

import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

# Kyutai clones whatever voice it is handed a sample of, so the sample IS the personality. The
# Expresso set ships the same speaker (ex03) acted in a dozen moods; the candidates are compared
# by scripts/bench_voice_style.py.
DEFAULT_VOICE = os.environ.get("MECHANIC_VOICE") or "expresso/ex03-ex01_happy_001_channel1_334s.wav"
# Sampling temperature for the audio tokens: lower means flatter, more predictable delivery.
DEFAULT_TEMP = float(os.environ.get("MECHANIC_VOICE_TEMP") or 0.6)
SAMPLE_RATE = 24000


@dataclass
class Speech:
    """One synthesised sentence and what it cost."""

    audio: "object"  # np.ndarray, typed loosely so the module imports without numpy
    sample_rate: int
    first_frame_ms: float | None
    total_ms: float
    frames: int = 0
    # The frames exactly as they were produced, kept so a cached line can be replayed in the
    # same pieces the player expects.
    chunks: list = field(default_factory=list, repr=False)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


@dataclass
class Synthesiser:
    """Loads on first use; the load costs seconds, so call `warm_up` at boot, never on a driver."""

    voice: str = DEFAULT_VOICE
    device: str = "cuda"
    n_q: int = 32
    temp: float = DEFAULT_TEMP
    cfg_coef: float = 2.0
    _tts: object | None = field(default=None, repr=False)
    _cond: object | None = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def _load(self):
        if self._tts is None:
            import torch
            from moshi.models.loaders import CheckpointInfo
            from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

            info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
            self._tts = TTSModel.from_checkpoint_info(
                info, n_q=self.n_q, temp=self.temp, device=torch.device(self.device)
            )
            self._cond = self._tts.make_condition_attributes(
                [self._tts.get_voice_path(self.voice)], cfg_coef=self.cfg_coef
            )
        return self._tts

    @property
    def sample_rate(self) -> int:
        """The codec's rate. Known for certain only once loaded; 24 kHz is what Kyutai uses."""
        return self._tts.mimi.sample_rate if self._tts is not None else SAMPLE_RATE

    def warm_up(self) -> None:
        """First synthesis pays CUDA warm-up — about six seconds. Spend it at startup."""
        self.say("Ready.")

    def prime(self, texts: Iterable[str]) -> None:
        """Synthesise ahead of time the lines the agent says word for word.

        The filler before a lookup, the safety warning, the apology for an empty turn: a handful
        of fixed strings, and usually the *first* thing the driver hears in a turn. Measured on
        2026-09-20, sound started about 650 ms after the sentence was ready — the synthesiser
        sharing the card with the language model, which is generating at the same time. From the
        cache that wait is gone, and it is the same voice saying the same words, so nothing about
        the answer changes.
        """
        self._load()
        for text in texts:
            if text not in self._cache:
                self._cache[text] = self.say(text, use_cache=False).chunks

    def say(self, text: str, on_frame: Callable[[object], None] | None = None, use_cache: bool = True) -> Speech:
        """Synthesise one sentence, handing each audio frame to `on_frame` as it appears."""
        import numpy as np

        if use_cache and (cached := self._cache.get(text)) is not None:
            for chunk in cached:
                if on_frame is not None:
                    on_frame(chunk)
            audio = np.concatenate(cached, axis=-1) if cached else np.zeros(0, dtype="float32")
            return Speech(audio, self.sample_rate, first_frame_ms=0.0, total_ms=0.0, frames=len(cached), chunks=cached)

        tts = self._load()
        entries = tts.prepare_script([text], padding_between=1)
        pcms: list = []
        first: float | None = None
        started = time.monotonic()

        def collect(frame):
            nonlocal first
            if (frame != -1).all():
                pcm = tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                chunk = np.clip(pcm[0, 0], -1, 1)
                if first is None:
                    first = (time.monotonic() - started) * 1000
                pcms.append(chunk)
                if on_frame is not None:
                    on_frame(chunk)

        with tts.mimi.streaming(1):
            tts.generate([entries], [self._cond], on_frame=collect)

        audio = np.concatenate(pcms, axis=-1) if pcms else np.zeros(0, dtype="float32")
        return Speech(
            audio=audio,
            sample_rate=tts.mimi.sample_rate,
            first_frame_ms=first,
            total_ms=(time.monotonic() - started) * 1000,
            frames=len(pcms),
            chunks=pcms,
        )

    def say_all(self, sentences: Iterator[str], on_frame: Callable[[object], None] | None = None) -> list[Speech]:
        return [self.say(s, on_frame) for s in sentences]
