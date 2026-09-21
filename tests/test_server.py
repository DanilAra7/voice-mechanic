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
