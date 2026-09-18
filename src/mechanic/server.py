"""FastAPI backend: Torque receiver + simulator control API.

uv run uvicorn mechanic.server:app --reload
"""

import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
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
