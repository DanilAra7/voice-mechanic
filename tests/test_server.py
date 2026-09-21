"""The gate in front of a published demo."""

from fastapi.testclient import TestClient


def test_an_open_url_is_strangers_on_our_gpu(monkeypatch):
    """One card runs one conversation. Without a gate the first passer-by takes the microphone
    away from the person we sent the link to."""
    import importlib

    monkeypatch.setenv("MECHANIC_ACCESS_KEY", "secret123")
    import mechanic.server as server

    importlib.reload(server)
    client = TestClient(server.app)

    assert client.get("/api/catalog").status_code == 401
    assert client.get("/api/health").status_code == 200, "our own health check must stay reachable"
    # A phone running Torque Pro has no cookie and no key: gating this endpoint told every
    # driver their adapter was unplugged, our own simulator included.
    assert client.get("/torque?id=demo&k5=95.0").status_code == 200

    opened = client.get("/api/catalog?k=secret123")
    assert opened.status_code == 200
    # The key was swapped for a cookie, so it is not re-sent on every later request.
    assert client.get("/api/catalog").status_code == 200

    monkeypatch.delenv("MECHANIC_ACCESS_KEY")
    importlib.reload(server)


def test_latency_is_kept_rather_than_remembered(tmp_path, monkeypatch):
    """The panel shows one session. Somebody closing the tab should not lose the measurement."""
    import mechanic.voice.ws as ws

    monkeypatch.setattr(ws, "LATENCY_LOG", tmp_path / "lat.jsonl")
    for ms in (700, 900, 1100, 1300, 2500):
        ws.record_client_latency({"client_ms": ms, "rtt_ms": 250}, session_id="demo")
    # A clock that jumped, and a message with no number in it, are not measurements.
    ws.record_client_latency({"client_ms": -5}, session_id="demo")
    ws.record_client_latency({"rtt_ms": 250}, session_id="demo")

    out = ws.latency_summary()
    assert out["turns"] == 5
    assert out["client_p50_ms"] == 1100
    assert out["client_min_ms"] == 700 and out["client_max_ms"] == 2500
    assert out["rtt_p50_ms"] == 250


def test_the_browsers_number_is_filed_with_the_servers_stages(tmp_path, monkeypatch):
    """A single figure says the wait was long, not which part of it was."""
    import mechanic.voice.ws as ws
    from mechanic.voice.session import TurnTimings

    monkeypatch.setattr(ws, "LATENCY_LOG", tmp_path / "lat.jsonl")
    stages = TurnTimings(asr_ms=300.4, first_sentence_ms=800.2, first_audio_ms=860.0, total_ms=4000.0)
    stages.tools = ["read_live_data"]
    for waited in (1500, 1600, 1700):
        ws.record_client_latency({"client_ms": waited, "rtt_ms": 60}, session_id="demo", server=stages)

    out = ws.latency_summary()
    assert out["recognition_ms"] == 300
    assert out["model_ms"] == 500
    assert out["synthesis_ms"] == 60
    assert out["server_total_ms"] == 860
    # The remainder is named, not left for somebody to subtract and guess at.
    assert out["network_and_browser_ms"] == out["client_p50_ms"] - 860
