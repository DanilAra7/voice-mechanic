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
