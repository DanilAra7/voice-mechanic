"""Stand-ins for the speech models, shared by the session and socket tests."""

import numpy as np

from mechanic.voice.asr import Transcript
from mechanic.voice.tts import Speech


class FakeRecognizer:
    def __init__(self, text="my idle feels rough"):
        self.text = text

    def warm_up(self):
        pass

    def transcribe(self, audio, sample_rate=16000):
        return Transcript(text=self.text, audio_seconds=len(audio) / 16000, decode_ms=10.0)


class FakeSynthesiser:
    sample_rate = 24000

    def __init__(self, frames=3):
        self.frames = frames
        self.said = []

    def warm_up(self):
        pass

    def say(self, text, on_frame=None):
        self.said.append(text)
        chunks = []
        for _ in range(self.frames):
            chunk = np.zeros(480, dtype=np.float32)
            chunks.append(chunk)
            if on_frame:
                on_frame(chunk)
        return Speech(audio=np.concatenate(chunks), sample_rate=24000, first_frame_ms=5.0, total_ms=20.0)
