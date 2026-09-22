"""Turn detection: silence must not become an utterance, and speech must come back whole."""

import numpy as np
import pytest
import soundfile as sf

from mechanic.data.common import ROOT
from mechanic.voice.asr import resample, to_mono
from mechanic.voice.vad import DEFAULT_MODEL, SAMPLE_RATE, TurnDetector

CLIP = ROOT / "data" / "cache" / "tts_samples" / "kyutai" / "00.wav"
pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="VAD model not downloaded")


def feed(detector, audio, chunk_s=0.1):
    step = int(SAMPLE_RATE * chunk_s)
    return [u for i in range(0, len(audio), step) for u in detector.push(audio[i : i + step])]


def test_silence_yields_nothing():
    assert feed(TurnDetector(), np.zeros(SAMPLE_RATE * 2, dtype=np.float32)) == []


def test_missing_model_says_where_to_look():
    with pytest.raises(FileNotFoundError, match="NOTES"):
        list(TurnDetector(model_path="/nonexistent/vad.onnx").push(np.zeros(1600, dtype=np.float32)))


@pytest.mark.skipif(not CLIP.exists(), reason="no sample audio")
def test_speech_then_silence_produces_one_utterance():
    audio, sr = sf.read(CLIP, dtype="float32")
    speech = resample(to_mono(audio), sr, SAMPLE_RATE)
    padded = np.concatenate(
        [np.zeros(SAMPLE_RATE // 2, dtype=np.float32), speech, np.zeros(SAMPLE_RATE, dtype=np.float32)]
    )
    utterances = feed(TurnDetector(), padded)
    assert len(utterances) == 1
    # Roughly the original speech, not the padding around it.
    assert 0.6 * len(speech) < len(utterances[0].audio) < 1.4 * len(speech)


@pytest.mark.skipif(not CLIP.exists(), reason="no sample audio")
def test_the_utterance_is_still_transcribable():
    """A clipped or misaligned segment would silently wreck recognition downstream."""
    from mechanic.voice.asr import DEFAULT_MODEL_DIR, Recognizer

    if not DEFAULT_MODEL_DIR.exists():
        pytest.skip("ASR model not downloaded")
    audio, sr = sf.read(CLIP, dtype="float32")
    speech = resample(to_mono(audio), sr, SAMPLE_RATE)
    padded = np.concatenate(
        [np.zeros(SAMPLE_RATE // 2, dtype=np.float32), speech, np.zeros(SAMPLE_RATE, dtype=np.float32)]
    )
    utterance = feed(TurnDetector(), padded)[0]
    assert "stop driving" in Recognizer().transcribe(utterance.audio).text.lower()


@pytest.mark.skipif(not CLIP.exists(), reason="no sample audio")
def test_the_silence_waited_out_is_reported():
    """Hands-free waits for the pause to prove the turn ended, and that wait is real latency.

    Every clock in the pipeline starts when the detector speaks up, which is already half a
    second after the driver's last word. Unless the utterance carries that wait, the server
    honestly reports a number that begins after the most expensive stage of the turn.
    """
    audio, sr = sf.read(CLIP, dtype="float32")
    speech = resample(to_mono(audio), sr, SAMPLE_RATE)
    padded = np.concatenate([speech, np.zeros(SAMPLE_RATE, dtype=np.float32)])

    utterance = feed(TurnDetector(min_silence_s=0.35), padded)[0]

    # The detector needs its silence threshold plus its own windowing before it will commit.
    assert 0.3 < utterance.trailing_s < 0.9, utterance.trailing_s


@pytest.mark.skipif(not CLIP.exists(), reason="no sample audio")
def test_a_declared_end_of_turn_waits_for_nothing():
    """The whole point of push-to-talk: the driver says they finished, so nothing is waited out."""
    audio, sr = sf.read(CLIP, dtype="float32")
    speech = resample(to_mono(audio), sr, SAMPLE_RATE)

    detector = TurnDetector()
    feed(detector, speech)  # no trailing silence, so nothing has been yielded yet
    flushed = list(detector.flush())

    assert flushed and flushed[0].trailing_s == 0.0
