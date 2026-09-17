import asyncio
import statistics

import httpx
import pytest
from fastapi import FastAPI

from mechanic.torque.receiver import build_router, parse_upload
from mechanic.torque.simulator import FAULTS, TorqueSimulator, VehicleModel, build_query
from mechanic.torque.store import TorqueStore
from mechanic.vehicles import VEHICLES, get_vehicle


def run(model: VehicleModel, seconds: int) -> list[dict[int, float]]:
    samples = []
    for _ in range(seconds):
        model.step(1.0)
        samples.append(model.state.as_pid_values())
    return samples


def test_parse_upload_values_and_meta():
    params = {
        "id": "phone1",
        "time": "1700000000000",
        "k5": "92.5",
        "kc": "750",
        "kff1005": "13.4",
        "userFullName5": "Engine Coolant Temperature",
        "userUnit5": "\\xC2\\xB0C",
        "defaultUnit5": "C",
        "kbad": "not-a-number",
    }
    device, ts, values, meta = parse_upload(params)
    assert device == "phone1"
    assert ts == 1700000000000
    assert values == {0x05: 92.5, 0x0C: 750.0, 0xFF1005: 13.4}
    assert meta[5] == {"full_name": "Engine Coolant Temperature", "unit": "°C"}


@pytest.mark.parametrize("vehicle_id", list(VEHICLES))
@pytest.mark.parametrize("mode", ["idle", "city", "highway"])
def test_healthy_car_is_plausible(vehicle_id, mode):
    v = get_vehicle(vehicle_id)
    model = VehicleModel(v, mode=mode, seed=1)
    samples = run(model, 300)
    for s in samples:
        assert 0 <= s[0x0C] <= v.redline_rpm
        assert 13.5 <= s[0x42] <= 14.6
        assert abs(s[0x06]) < 12
        assert 0 <= s[0x2F] <= 100
    assert abs(samples[-1][0x05] - v.coolant_target_c) < 4
    assert model.active_dtcs() == []


def test_coolant_leak_overheats():
    model = VehicleModel(get_vehicle("audi_a4_b8"), mode="idle", fault="coolant_leak", seed=1)
    samples = run(model, 600)
    assert samples[-1][0x05] > 115
    assert "P0217" in model.active_dtcs()


def test_thermostat_stuck_open_runs_cold():
    model = VehicleModel(get_vehicle("honda_civic_10"), mode="highway", fault="thermostat_stuck_open", seed=1)
    samples = run(model, 900)
    assert samples[-1][0x05] < 70
    assert model.active_dtcs() == ["P0128"]


def test_vacuum_leak_trims_high_at_idle_low_at_cruise():
    idle = VehicleModel(get_vehicle("audi_a4_b8"), mode="idle", fault="vacuum_leak", seed=1)
    cruise = VehicleModel(get_vehicle("audi_a4_b8"), mode="highway", fault="vacuum_leak", seed=1)
    idle_stft = statistics.mean(s[0x06] for s in run(idle, 200)[-60:])
    cruise_stft = statistics.mean(s[0x06] for s in run(cruise, 200)[-60:])
    assert idle_stft > 12
    assert cruise_stft < idle_stft / 2
    assert idle.active_dtcs() == ["P0171"]


def test_misfire_makes_idle_rpm_unstable():
    healthy = VehicleModel(get_vehicle("toyota_corolla_11"), mode="idle", seed=1)
    faulty = VehicleModel(get_vehicle("toyota_corolla_11"), mode="idle", fault="misfire_cyl1", seed=1)
    assert statistics.pstdev(s[0x0C] for s in run(faulty, 120)) > 3 * statistics.pstdev(
        s[0x0C] for s in run(healthy, 120)
    )
    assert faulty.active_dtcs() == ["P0300", "P0301"]


def test_failing_alternator_drops_voltage():
    model = VehicleModel(get_vehicle("ford_f150_13"), mode="city", fault="failing_alternator", seed=1)
    samples = run(model, 400)
    assert samples[-1][0x42] < 12.2
    assert model.active_dtcs() == ["P0562"]


def test_every_fault_eventually_sets_a_code():
    for fault in FAULTS:
        model = VehicleModel(get_vehicle("honda_accord_9"), mode="idle", fault=fault, seed=2)
        run(model, 900)
        assert model.active_dtcs(), fault


def test_query_roundtrip_through_parser():
    model = VehicleModel(get_vehicle("audi_a4_b8"), seed=1)
    model.step()
    values = model.state.as_pid_values()
    params = build_query("dev", "123", "a@b.c", 42, values, include_meta=True)
    device, ts, parsed, meta = parse_upload(params)
    assert (device, ts) == ("dev", 42)
    assert parsed == {k: float(v) for k, v in values.items()}
    assert meta[0x05]["unit"] == "°C"


async def test_simulator_uploads_over_http_to_receiver():
    store = TorqueStore()
    app = FastAPI()
    app.include_router(build_router(store))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        model = VehicleModel(get_vehicle("audi_a4_b8"), mode="idle", fault="vacuum_leak", seed=1)
        sim = TorqueSimulator(
            model,
            "http://test/torque",
            device="sim-test",
            interval_s=0.01,
            time_scale=1000,
            on_dtcs=store.set_dtcs,
            client=client,
        )
        sim.start()

        await asyncio.sleep(0.5)
        await sim.stop()

    assert sim.uploads > 10 and sim.last_error is None
    latest = {r.pid: r for r in store.latest("sim-test")}
    assert latest[0x05].unit == "°C" and latest[0x05].name == "Engine Coolant Temperature"
    assert store.get_dtcs("sim-test") == ["P0171"]
    trend = store.trend("sim-test", 0x06, window_s=60)
    assert trend is not None and trend.samples > 5
