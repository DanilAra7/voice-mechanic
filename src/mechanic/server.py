"""FastAPI backend: Torque receiver + simulator control API.

uv run uvicorn mechanic.server:app --reload
"""

import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mechanic.config import load_env
from mechanic.torque.receiver import build_router
from mechanic.torque.simulator import DRIVE_MODES, FAULTS, TorqueSimulator, VehicleModel
from mechanic.torque.store import TorqueStore
from mechanic.vehicles import VEHICLES, get_vehicle
from mechanic.voice.ws import SharedModels
from mechanic.voice.ws import build_router as build_voice_router

load_env()

DB_PATH = os.environ.get("MECHANIC_DB", "data/cache/torque.sqlite")
# Where /app/record.html puts the clips it captures. Writing files from a web request is not
# something a published demo should offer, so it stays off unless somebody asks for it.
RECORDING_DIR = Path(__file__).resolve().parents[2] / "evals" / "audio" / "real"
RECORDING_ENABLED = os.environ.get("MECHANIC_ALLOW_RECORDING") == "1"
RECORDING_NAME = re.compile(r"^\d{2}\.wav$")
MAX_RECORDING_BYTES = 8 * 1024 * 1024
# The simulator talks to our own receiver over real HTTP, exactly like a phone would.
SELF_URL = os.environ.get("MECHANIC_SELF_URL", "http://127.0.0.1:8000")


class SimulatorManager:
    def __init__(self, store: TorqueStore, torque_url: str):
        self.store = store
        self.torque_url = torque_url
        self.sims: dict[str, TorqueSimulator] = {}

    async def start(
        self, device: str, vehicle_id: str, mode: str, fault: str | None, time_scale: float
    ) -> TorqueSimulator:
        await self.stop(device)
        self.store.clear_device(device)
        model = VehicleModel(get_vehicle(vehicle_id), mode=mode, fault=fault)
        sim = TorqueSimulator(model, self.torque_url, device=device, time_scale=time_scale, on_dtcs=self.store.set_dtcs)
        sim.start()
        self.sims[device] = sim
        return sim

    async def stop(self, device: str) -> None:
        if sim := self.sims.pop(device, None):
            await sim.stop()

    async def stop_all(self) -> None:
        for device in list(self.sims):
            await self.stop(device)

    def get(self, device: str) -> TorqueSimulator:
        try:
            return self.sims[device]
        except KeyError:
            raise HTTPException(404, f"No simulator running for device '{device}'") from None


store = TorqueStore(DB_PATH)
simulators = SimulatorManager(store, f"{SELF_URL}/torque")
# Speech models are heavy: one copy per process, loaded lazily so the API still starts without them.
voice_models = SharedModels()
# Last time a human did something here. scripts/idle_shutdown.py polls this to stop a rented GPU.
_last_activity = time.time()


def mark_activity() -> None:
    global _last_activity
    _last_activity = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await simulators.stop_all()


app = FastAPI(title="Voice Mechanic", lifespan=lifespan)


class AccessKey:
    """A shared secret on the link, because an open URL is strangers on our GPU.

    One card runs one conversation. A public address with no gate is not a demo, it is an
    invitation, and the first person to find it takes the microphone away from the person we
    sent the link to. This is deliberately the weakest thing that works: a secret in the query
    string, swapped for a cookie on the first request so it is not re-sent on every later one.
    It keeps out passers-by. It is not authentication and is not pretending to be - the demo
    holds no accounts and no personal data, and when MECHANIC_ACCESS_KEY is unset (local work)
    nothing is gated at all.
    """

    COOKIE = "mechanic_key"

    def __init__(self, app, key: str):
        self.app = app
        self.key = key

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("path") == "/api/health":
            await self.app(scope, receive, send)
            return

        query = parse_qs(scope.get("query_string", b"").decode())
        headers = dict(scope.get("headers") or [])
        cookies = SimpleCookie(headers.get(b"cookie", b"").decode())
        from_query = (query.get("k") or [""])[0]
        from_cookie = cookies[self.COOKIE].value if self.COOKIE in cookies else ""

        if from_query == self.key or from_cookie == self.key:
            if from_query == self.key and scope["type"] == "http":
                await self.app(scope, receive, self._set_cookie(send))
            else:
                await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await PlainTextResponse("This demo is open by link only.", status_code=401)(scope, receive, send)

    def _set_cookie(self, send):
        async def wrapped(message):
            if message["type"] == "http.response.start":
                cookie = f"{self.COOKIE}={self.key}; Path=/; Max-Age=86400; SameSite=Lax"
                message.setdefault("headers", []).append((b"set-cookie", cookie.encode()))
            await send(message)

        return wrapped


if ACCESS_KEY := os.environ.get("MECHANIC_ACCESS_KEY"):
    app.add_middleware(AccessKey, key=ACCESS_KEY)

# The deployed front end is static and hosted separately; serving it here keeps local testing
# to one origin, which is also the only way the browser hands over a microphone without HTTPS.
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
if WEB_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")
app.include_router(build_router(store))
app.include_router(build_voice_router(store, voice_models))


class StartSimRequest(BaseModel):
    vehicle: str = "audi_a4_b8"
    mode: str = "city"
    fault: str | None = None
    time_scale: float = 1.0


class UpdateSimRequest(BaseModel):
    mode: str | None = None
    fault: str | None = None
    clear_fault: bool = False


def sim_status(sim: TorqueSimulator) -> dict:
    m = sim.model
    return {
        "device": sim.device,
        "vehicle": m.vehicle.id,
        "mode": m.mode,
        "fault": m.fault,
        "sim_time_s": round(m.sim_time_s),
        "fault_time_s": round(m.fault_time_s),
        "uploads": sim.uploads,
        "last_error": sim.last_error,
        "dtcs": m.active_dtcs(),
    }


@app.put("/api/recording/{name}")
async def save_recording(name: str, request: Request) -> dict:
    """Take a clip straight from the browser instead of relying on a download.

    The download path failed silently on 2026-09-20 — fifteen takes recorded, nothing on disk
    anywhere — and a recording nobody can find is worse than no recording, because it is only
    discovered after the reading is done.
    """
    if not RECORDING_ENABLED:
        raise HTTPException(403, "recording is off; start the server with MECHANIC_ALLOW_RECORDING=1")
    if not RECORDING_NAME.match(name):
        raise HTTPException(400, "name must look like 07.wav")
    body = await request.body()
    if not body or len(body) > MAX_RECORDING_BYTES:
        raise HTTPException(400, f"clip is {len(body)} bytes")
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    (RECORDING_DIR / name).write_bytes(body)
    return {"saved": name, "bytes": len(body), "path": str(RECORDING_DIR / name)}


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "seconds_since_activity": round(time.time() - _last_activity, 1),
        "simulators": len(simulators.sims),
    }


@app.get("/api/catalog")
async def catalog() -> dict:
    return {
        "vehicles": [{**asdict(v), "title": v.title} for v in VEHICLES.values()],
        "modes": list(DRIVE_MODES),
        "faults": [asdict(f) for f in FAULTS.values()],
    }


@app.post("/api/sim/{device}")
async def start_sim(device: str, req: StartSimRequest) -> dict:
    mark_activity()
    try:
        sim = await simulators.start(device, req.vehicle, req.mode, req.fault, req.time_scale)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from None
    return sim_status(sim)


@app.patch("/api/sim/{device}")
async def update_sim(device: str, req: UpdateSimRequest) -> dict:
    mark_activity()
    sim = simulators.get(device)
    try:
        if req.mode:
            sim.model.set_mode(req.mode)
        if req.clear_fault:
            sim.model.set_fault(None)
        elif req.fault:
            sim.model.set_fault(req.fault)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return sim_status(sim)


@app.delete("/api/sim/{device}")
async def stop_sim(device: str) -> dict:
    await simulators.stop(device)
    return {"stopped": device}


@app.get("/api/sim/{device}")
async def get_sim(device: str) -> dict:
    return sim_status(simulators.get(device))


@app.get("/api/sensors/{device}")
async def sensors(device: str) -> dict:
    mark_activity()
    return {
        "device": device,
        "readings": [asdict(r) for r in store.latest(device)],
        "dtcs": store.get_dtcs(device),
    }
