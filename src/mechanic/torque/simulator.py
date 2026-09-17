"""Simulated car that streams sensor data using the Torque Pro upload protocol.

The model is deliberately simple ("physics-lite"): enough for the numbers to be
plausible and for each injected fault to leave a recognisable signature that the
agent has to reason about (fuel trims at idle vs. cruise, temperature trends,
charging voltage, ...). Faults progress over simulated time.

Run standalone against any Torque-compatible endpoint:

    uv run python -m mechanic.torque.simulator --url http://127.0.0.1:8000/torque \
        --vehicle audi_a4_b8 --mode city --fault vacuum_leak
"""

import argparse
import asyncio
import math
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from mechanic.torque.pids import PIDS
from mechanic.vehicles import VEHICLES, Vehicle, get_vehicle

AMBIENT_C = 22.0
DRIVE_MODES = ("idle", "city", "highway")


@dataclass
class EngineState:
    speed_kmh: float = 0.0
    rpm: float = 0.0
    throttle_pct: float = 0.0
    load_pct: float = 0.0
    maf_gs: float = 0.0
    map_kpa: float = 0.0
    timing_deg: float = 0.0
    coolant_c: float = AMBIENT_C
    oil_c: float = AMBIENT_C
    iat_c: float = AMBIENT_C
    stft_pct: float = 0.0
    ltft_pct: float = 0.0
    fuel_pct: float = 68.0
    volts: float = 12.6

    def as_pid_values(self) -> dict[int, float]:
        return {
            0x04: round(self.load_pct, 1),
            0x05: round(self.coolant_c, 1),
            0x06: round(self.stft_pct, 1),
            0x07: round(self.ltft_pct, 1),
            0x0B: round(self.map_kpa, 1),
            0x0C: round(self.rpm),
            0x0D: round(self.speed_kmh),
            0x0E: round(self.timing_deg, 1),
            0x0F: round(self.iat_c, 1),
            0x10: round(self.maf_gs, 2),
            0x11: round(self.throttle_pct, 1),
            0x2F: round(self.fuel_pct, 1),
            0x42: round(self.volts, 2),
            0x46: AMBIENT_C,
            0x5C: round(self.oil_c, 1),
        }


@dataclass(frozen=True)
class Fault:
    id: str
    title: str
    # What is physically wrong. Shown in the Garage UI only — never to the agent.
    hidden_cause: str
    symptom_hint: str


FAULTS: dict[str, Fault] = {
    f.id: f
    for f in [
        Fault(
            "coolant_leak",
            "Overheating",
            "Coolant leak: the level drops over time, temperature climbs, worst at idle.",
            "Temperature gauge creeping up, sweet smell after driving.",
        ),
        Fault(
            "thermostat_stuck_open",
            "Engine never warms up",
            "Thermostat stuck open: coolant stays well below operating temperature, colder on the highway.",
            "Heater blows lukewarm air, fuel economy got worse.",
        ),
        Fault(
            "misfire_cyl1",
            "Rough idle / misfire",
            "Cylinder 1 misfire (e.g. worn coil pack or spark plug): shaky RPM at idle.",
            "Car shakes at stop lights, check engine light flashing under load.",
        ),
        Fault(
            "vacuum_leak",
            "Lean condition (vacuum leak)",
            "Vacuum leak (cracked PCV hose/valve): high positive fuel trims at idle that shrink at cruise.",
            "Hissing sound from the engine bay, slightly high idle.",
        ),
        Fault(
            "dirty_maf",
            "Dirty MAF sensor",
            "Contaminated MAF sensor under-reports airflow: positive fuel trims at all loads.",
            "Hesitation when accelerating, poor fuel economy.",
        ),
        Fault(
            "failing_alternator",
            "Charging system failure",
            "Alternator is failing: system voltage falls while the engine runs.",
            "Battery light flickers, headlights dim at idle.",
        ),
    ]
}


@dataclass
class VehicleModel:
    vehicle: Vehicle
    mode: str = "city"
    fault: str | None = None
    cold_start: bool = False
    seed: int | None = None
    state: EngineState = field(default_factory=EngineState)
    sim_time_s: float = 0.0
    fault_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in DRIVE_MODES:
            raise ValueError(f"mode must be one of {DRIVE_MODES}")
        self.set_fault(self.fault)
        self._rng = random.Random(self.seed)
        s = self.state
        s.rpm = self.vehicle.idle_rpm
        warm = not self.cold_start
        s.coolant_c = self.vehicle.coolant_target_c - 2 if warm else AMBIENT_C
        s.oil_c = s.coolant_c - 3 if warm else AMBIENT_C
        s.ltft_pct = self._rng.uniform(-2.5, 2.5)
        s.volts = 14.1

    def set_fault(self, fault: str | None) -> None:
        if fault is not None and fault not in FAULTS:
            raise ValueError(f"Unknown fault '{fault}'. Known: {', '.join(FAULTS)}")
        self.fault = fault
        self.fault_time_s = 0.0

    def set_mode(self, mode: str) -> None:
        if mode not in DRIVE_MODES:
            raise ValueError(f"mode must be one of {DRIVE_MODES}")
        self.mode = mode

    # --- driving profile -------------------------------------------------

    def _target_speed(self) -> float:
        t = self.sim_time_s
        if self.mode == "idle":
            return 0.0
        if self.mode == "highway":
            return 110 + 6 * math.sin(t / 45)
        # City: 90 s cycle — stopped 20 s, accelerate, cruise ~50 km/h, brake.
        phase = t % 90
        if phase < 20:
            return 0.0
        if phase < 75:
            return 48 + 6 * math.sin(t / 9)
        return 0.0

    # --- one simulation step --------------------------------------------

    def step(self, dt: float = 1.0) -> EngineState:
        v, s, rng = self.vehicle, self.state, self._rng
        self.sim_time_s += dt
        if self.fault:
            self.fault_time_s += dt
        ft = self.fault_time_s

        # Speed follows the profile with limited acceleration/braking.
        target = self._target_speed()
        dv = target - s.speed_kmh
        accel = max(-12.0, min(8.0, dv)) * dt
        s.speed_kmh = max(0.0, s.speed_kmh + accel)
        accelerating = accel > 1.0
        idling = s.speed_kmh < 3

        # RPM / throttle / load.
        if idling:
            rpm = v.idle_rpm + rng.gauss(0, 8)
            throttle, load = 13.0, 21.0
        else:
            ratio = 20.0 if s.speed_kmh > 80 else 34.0
            rpm = max(v.idle_rpm * 1.4, s.speed_kmh * ratio)
            throttle = 18 + (35 if accelerating else 6) + rng.gauss(0, 1.5)
            load = 35 + (38 if accelerating else 4) + rng.gauss(0, 2)
        if accelerating:
            rpm += 600
        s.rpm = min(rpm, v.redline_rpm)
        s.throttle_pct = throttle
        s.load_pct = max(0.0, min(100.0, load))

        # Air flow, manifold pressure, timing.
        air_factor = 0.15 + s.load_pct / 100 * 0.85
        true_maf = v.displacement_l * s.rpm * air_factor * 0.0045
        s.maf_gs = true_maf
        if v.turbo and s.load_pct > 60:
            s.map_kpa = 100 + (s.load_pct - 60) * 2.2
        else:
            s.map_kpa = min(100.0, 28 + s.load_pct * 0.7)
        s.timing_deg = (12 if idling else 30 - s.load_pct * 0.15) + rng.gauss(0, 0.8)

        # Fuel trims: small noise around a slightly-off long-term trim.
        s.stft_pct = rng.gauss(0, 1.8)

        # Thermal model: coolant approaches the thermostat set-point.
        coolant_target = v.coolant_target_c
        cooling_time_constant = 150.0

        # Charging voltage.
        s.volts = 14.1 + rng.gauss(0, 0.05)

        # --- faults ------------------------------------------------------
        if self.fault == "coolant_leak":
            # Effective set-point rises as coolant is lost; airflow helps a bit.
            coolant_target = v.coolant_target_c + min(30.0, ft * 0.12) - (4 if s.speed_kmh > 60 else 0)
            cooling_time_constant = 90.0
        elif self.fault == "thermostat_stuck_open":
            coolant_target = AMBIENT_C + (38 if s.speed_kmh > 80 else 48)
        elif self.fault == "misfire_cyl1":
            if idling:
                s.rpm += rng.gauss(0, 55) - 25
                s.load_pct += rng.gauss(0, 4)
            s.stft_pct += 3.5
            s.timing_deg += rng.gauss(0, 2.5)
        elif self.fault == "vacuum_leak":
            severity = min(1.0, 0.4 + ft / 120)
            # Unmetered air matters most when real airflow is small (idle).
            share = 1.0 if idling else max(0.2, 1 - s.load_pct / 60)
            s.stft_pct += 17 * severity * share
            s.ltft_pct = min(s.ltft_pct + 0.05 * dt, 9.0) if idling else s.ltft_pct
            if idling:
                s.rpm += 70 * severity
        elif self.fault == "dirty_maf":
            s.maf_gs = true_maf * 0.72
            s.stft_pct += 9
            s.ltft_pct = min(s.ltft_pct + 0.08 * dt, 13.0)
        elif self.fault == "failing_alternator":
            s.volts = max(11.4, 13.4 - ft * 0.006) + rng.gauss(0, 0.04)

        a = min(1.0, dt / cooling_time_constant)
        s.coolant_c += (coolant_target - s.coolant_c) * a + rng.gauss(0, 0.1)
        s.oil_c += ((s.coolant_c + (6 if s.speed_kmh > 80 else 2)) - s.oil_c) * min(1.0, dt / 240)
        s.iat_c = AMBIENT_C + (9 if idling else 3) + rng.gauss(0, 0.3)
        # Petrol: stoichiometric AFR 14.7, density ~740 g/L.
        s.fuel_pct = max(0.0, s.fuel_pct - true_maf * dt / 14.7 / 740 / v.fuel_tank_l * 100)
        s.rpm = max(0.0, s.rpm)
        return s

    def active_dtcs(self) -> list[str]:
        s, ft = self.state, self.fault_time_s
        match self.fault:
            case "coolant_leak":
                return ["P0217"] if s.coolant_c > 115 else []
            case "thermostat_stuck_open":
                return ["P0128"] if ft > 60 else []
            case "misfire_cyl1":
                return ["P0300", "P0301"] if ft > 20 else []
            case "vacuum_leak":
                return ["P0171"] if ft > 45 else []
            case "dirty_maf":
                return ["P0101", "P0171"] if ft > 60 else []
            case "failing_alternator":
                return ["P0562"] if s.volts < 12.2 else []
        return []


def build_query(
    device: str, session: str, email: str, ts_ms: int, values: dict[int, float], include_meta: bool
) -> dict[str, str]:
    params = {"v": "8", "session": session, "id": device, "eml": email, "time": str(ts_ms)}
    for pid, value in values.items():
        params[PIDS[pid].torque_key] = str(value)
        if include_meta:
            hexpid = f"{pid:x}"
            params[f"userFullName{hexpid}"] = PIDS[pid].full_name
            params[f"userShortName{hexpid}"] = PIDS[pid].short_name
            params[f"userUnit{hexpid}"] = PIDS[pid].unit
            params[f"defaultUnit{hexpid}"] = PIDS[pid].unit
    return params


class TorqueSimulator:
    """Steps a VehicleModel in real time and uploads each sample over HTTP."""

    def __init__(
        self,
        model: VehicleModel,
        url: str,
        device: str | None = None,
        email: str = "simulator@example.com",
        interval_s: float = 1.0,
        time_scale: float = 1.0,
        on_dtcs: Callable[[str, list[str]], None] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.url = url
        self.device = device or f"sim-{uuid.uuid4().hex[:8]}"
        self.email = email
        self.interval_s = interval_s
        self.time_scale = time_scale
        self.on_dtcs = on_dtcs
        self.session = str(int(time.time() * 1000))
        self._client = client
        self._task: asyncio.Task | None = None
        self.uploads = 0
        self.last_error: str | None = None

    async def run(self) -> None:
        client = self._client or httpx.AsyncClient(timeout=5)
        try:
            while True:
                started = time.monotonic()
                self.model.step(self.interval_s * self.time_scale)
                params = build_query(
                    self.device,
                    self.session,
                    self.email,
                    int(time.time() * 1000),
                    self.model.state.as_pid_values(),
                    include_meta=self.uploads % 30 == 0,
                )
                try:
                    r = await client.get(self.url, params=params)
                    r.raise_for_status()
                    self.uploads += 1
                    self.last_error = None
                except httpx.HTTPError as e:
                    self.last_error = str(e)
                if self.on_dtcs:
                    self.on_dtcs(self.device, self.model.active_dtcs())
                await asyncio.sleep(max(0.0, self.interval_s - (time.monotonic() - started)))
        finally:
            if self._client is None:
                await client.aclose()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000/torque")
    p.add_argument("--vehicle", default="audi_a4_b8", choices=list(VEHICLES))
    p.add_argument("--mode", default="city", choices=DRIVE_MODES)
    p.add_argument("--fault", default=None, choices=list(FAULTS))
    p.add_argument("--device", default=None)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--cold-start", action="store_true")
    args = p.parse_args()

    model = VehicleModel(get_vehicle(args.vehicle), mode=args.mode, fault=args.fault, cold_start=args.cold_start)
    last_codes: list[str] = []

    def print_dtcs(device: str, codes: list[str]) -> None:
        if codes != last_codes:
            print(f"[{device}] DTCs: {', '.join(codes) or 'none'}")
            last_codes[:] = codes

    sim = TorqueSimulator(model, args.url, device=args.device, time_scale=args.time_scale, on_dtcs=print_dtcs)
    print(f"Simulating {model.vehicle.title} as device '{sim.device}' → {args.url}")
    asyncio.run(sim.run())


if __name__ == "__main__":
    main()
