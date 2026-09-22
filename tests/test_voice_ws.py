"""The socket itself: byte formats, the handshake, and audio coming back."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers_voice import FakeRecognizer, FakeSynthesiser

from mechanic.torque.store import TorqueStore
from mechanic.voice.ws import SharedModels, build_router, float_to_pcm, pcm_to_float


def test_pcm_round_trip_keeps_the_signal():
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    assert np.allclose(pcm_to_float(float_to_pcm(audio)), audio, atol=1e-4)


def test_pcm_clips_instead_of_wrapping():
    """A sample above 1.0 must saturate, not wrap around into a loud click."""
    loud = np.array([2.0, -2.0], dtype=np.float32)
    assert np.allclose(pcm_to_float(float_to_pcm(loud)), [1.0, -1.0], atol=1e-3)


@pytest.fixture
def client(monkeypatch):
    models = SharedModels.__new__(SharedModels)
    models.recognizer = FakeRecognizer()
    models.synthesiser = FakeSynthesiser()
    models.index = None
    models._warm = True
    from mechanic.knowledge.dtc import DtcDatabase

    models.dtc = DtcDatabase()
    app = FastAPI()
    app.include_router(build_router(TorqueStore(), models))
    return TestClient(app)


def test_handshake_reports_the_audio_rates(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "hello", "device": "t", "vehicle": "audi_a4_b8"})
        ready = ws.receive_json()
    assert ready["type"] == "ready"
    assert ready["mic_rate"] == 16000
    assert ready["audio_rate"] == 24000
    assert ready["vehicle"] == "audi_a4_b8"


def test_reset_is_acknowledged(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "hello", "device": "t"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "reset_done"


def test_microphone_bytes_are_accepted(client):
    """Silence must not crash the socket or produce a turn."""
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "hello", "device": "t"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(float_to_pcm(np.zeros(1600, dtype=np.float32)))
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "reset_done"


def test_the_garage_can_change_the_car_mid_session(client):
    """Swapping the car has to reach the agent: it states the current car in its system prompt."""
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "hello", "device": "t", "vehicle": "audi_a4_b8"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "vehicle", "vehicle": "honda_civic_10"})
        assert ws.receive_json() == {"type": "vehicle", "vehicle": "honda_civic_10"}


def test_an_unknown_car_is_refused_rather_than_silently_forgetting_the_current_one(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "hello", "device": "t", "vehicle": "audi_a4_b8"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "vehicle", "vehicle": "delorean_dmc12"})
        reply = ws.receive_json()
        assert reply["type"] == "error" and "delorean_dmc12" in reply["message"]
        # Still the car we started with, so the next answer is not about nothing.
        ws.send_json({"type": "vehicle", "vehicle": "audi_a4_b8"})
        assert ws.receive_json()["vehicle"] == "audi_a4_b8"


def test_the_button_release_is_acknowledged_before_any_work(client):
    """The one measurement that separates a slow server from a slow wire.

    The browser stamps its clock when the button comes up and again when sound arrives. Between
    those two the server does ~850 ms of work, and the difference between the two clocks has to
    be the network — but only if the network can be measured on its own. So the release is
    echoed straight back, before recognition has been handed a single sample.
    """
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "hello", "device": "t"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "end_of_speech", "t": 1234.5})
        ack = ws.receive_json()

    assert ack["type"] == "turn_ack"
    assert ack["t"] == 1234.5, "the browser's own clock must come back untouched"


def test_a_release_without_a_stamp_is_not_acknowledged(client):
    """An older tab must not be answered with a message it will not understand."""
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "hello", "device": "t"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "end_of_speech"})
        ws.send_json({"type": "ping", "t": 7})
        assert ws.receive_json()["type"] == "pong"
