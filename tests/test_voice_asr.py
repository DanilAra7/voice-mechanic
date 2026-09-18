"""ASR wrapper: resampling maths and a real decode when the model is present."""

import numpy as np
import pytest

from mechanic.voice.asr import DEFAULT_MODEL_DIR, Recognizer, resample, to_mono


def test_resample_changes_length_and_keeps_shape():
    audio = np.sin(np.linspace(0, 20, 24000)).astype(np.float32)
    out = resample(audio, 24000, 16000)
    assert len(out) == 16000
    assert out.dtype == np.float32


def test_resample_is_a_no_op_at_the_target_rate():
    audio = np.zeros(100, dtype=np.float32)
    assert resample(audio, 16000, 16000) is not None
    assert len(resample(audio, 16000, 16000)) == 100


def test_to_mono_averages_channels():
    stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    assert to_mono(stereo).tolist() == [2.0, 3.0]


def test_missing_model_says_where_to_look():
    rec = Recognizer(model_dir="/nonexistent/asr")
    with pytest.raises(FileNotFoundError, match="NOTES"):
        rec.transcribe(np.zeros(1600, dtype=np.float32))


@pytest.mark.skipif(not DEFAULT_MODEL_DIR.exists(), reason="ASR model not downloaded")
def test_transcribes_a_real_clip():
    import soundfile as sf

    from mechanic.data.common import ROOT

    clip = ROOT / "data" / "cache" / "tts_samples" / "kyutai" / "00.wav"
    if not clip.exists():
        pytest.skip("no sample audio")
    audio, sr = sf.read(clip, dtype="float32")
    result = Recognizer().transcribe(to_mono(audio), sr)
    assert "stop driving" in result.text.lower()
    assert result.decode_ms < 2000
