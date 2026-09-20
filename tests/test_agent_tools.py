import time

import pytest
from conftest import FakeIndex

from mechanic.agent.tools import TOOL_NAMES, TOOL_SCHEMAS, Session, ToolRunner
from mechanic.knowledge.dtc import DtcDatabase, normalize_code
from mechanic.torque.simulator import VehicleModel
from mechanic.torque.store import TorqueStore
from mechanic.vehicles import get_vehicle


@pytest.fixture
def runner_and_session():
    store = TorqueStore()
    model = VehicleModel(get_vehicle("audi_a4_b8"), mode="idle", fault="coolant_leak", seed=1)
    now_ms = int(time.time() * 1000)
    for i in range(300):
        model.step(2.0)
        store.add_upload("dev1", now_ms - (300 - i) * 1000, model.state.as_pid_values())
    store.set_dtcs("dev1", model.active_dtcs())
    runner = ToolRunner(store, DtcDatabase(), FakeIndex())
    return runner, Session(device="dev1", vehicle_id="audi_a4_b8")


def test_tool_schemas_are_wellformed():
    assert len(TOOL_SCHEMAS) == len(set(TOOL_NAMES)) == 8
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        assert schema["type"] == "function"
        assert fn["description"] and len(fn["description"]) < 400
        params = fn["parameters"]
        assert params["type"] == "object"
        for req in params.get("required", []):
            assert req in params["properties"], fn["name"]


def test_unknown_tool_and_bad_arguments(runner_and_session):
    runner, session = runner_and_session
    assert "error" in runner.call("nope", {}, session)
    assert "error" in runner.call("lookup_dtc", {"wrong": "x"}, session)


def test_read_live_data_reports_values_and_codes(runner_and_session):
    runner, session = runner_and_session
    out = runner.call("read_live_data", {}, session)
    assert out["connected"]
    coolant = next(r for r in out["readings"] if "Coolant" in r["sensor"])
    assert coolant["value"] > 110 and coolant["unit"] == "°C"
    assert "overheating" in coolant["expected"]
    assert "P0217" in out["trouble_codes"]

    filtered = runner.call("read_live_data", {"sensors": ["coolant", "voltage"]}, session)
    assert {r["sensor"] for r in filtered["readings"]} == {"Engine Coolant Temperature", "Voltage (Control Module)"}


def test_read_live_data_without_adapter():
    runner = ToolRunner(TorqueStore(), DtcDatabase(), FakeIndex())
    out = runner.call("read_live_data", {}, Session(device="nobody"))
    assert out["connected"] is False


def test_sensor_trend_detects_rising_temperature(runner_and_session):
    runner, session = runner_and_session
    out = runner.call("get_sensor_trend", {"sensor": "coolant", "minutes": 5}, session)
    assert out["direction"] == "rising"
    assert out["change_per_minute"] > 0
    assert out["last"] > out["first"]
    assert "error" not in runner.call("get_sensor_trend", {"sensor": "volts"}, session)
    assert "error" in runner.call("get_sensor_trend", {"sensor": "flux capacitor"}, session)


def test_lookup_dtc_from_spoken_code(runner_and_session):
    runner, session = runner_and_session
    out = runner.call("lookup_dtc", {"code": "P zero one seven one"}, session)
    assert out["code"] == "P0171"
    assert "vacuum leak" in out["summary"].lower()
    assert "error" in runner.call("lookup_dtc", {"code": "banana"}, session)


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("P0171", "P0171"),
        ("p 0171", "P0171"),
        ("code P zero three zero one", "P0301"),
        ("P171", "P0171"),
        ("0171", "P0171"),
        ("U0100", "U0100"),
        ("hello there", None),
        ("", None),
    ],
)
def test_normalize_code(spoken, expected):
    assert normalize_code(spoken) == expected


def test_vehicle_tools(runner_and_session):
    runner, session = runner_and_session
    assert "A4" in runner.call("get_vehicle", {}, session)["vehicle"]

    empty = Session(device="dev1")
    assert runner.call("get_vehicle", {}, empty)["vehicle"] is None
    out = runner.call("set_vehicle", {"description": "I drive a 2018 Honda Civic"}, empty)
    assert empty.vehicle_id == "honda_civic_10" and "Civic" in out["vehicle"]

    old = runner.call("set_vehicle", {"description": "1999 Honda Civic"}, empty)
    assert "knowledge base" in old["message"]
    assert "error" in runner.call("set_vehicle", {"description": "Lada Niva"}, empty)


def test_search_tools_pass_source_and_vehicle(runner_and_session):
    runner, session = runner_and_session
    for tool, source in [
        ("search_forum", "stackexchange"),
        ("search_owner_reports", "startmycar"),
        ("search_how_to", "carcarekiosk"),
    ]:
        out = runner.call(tool, {"query": "coolant leak"}, session)
        assert runner.index.calls[-1] == ("coolant leak", (source,), "audi_a4_b8")
        assert len(out["results"][0]["text"]) <= 600


def test_a_failed_turn_says_whose_fault_it_was():
    """The eval runner is not imported anywhere else, so a broken dataclass in it only shows up
    on the rented GPU, half an hour and one instance start later."""
    from mechanic.evals.run import score_turn

    spec = {"user": "why are my trims high", "expect_any": [["vacuum leak"]]}
    tools = [{"name": "search_forum", "result": {"hits": [{"text": "this is a vacuum leak"}]}}]
    timings = {"first_token_ms": 1.0, "first_sentence_ms": 2.0, "total_ms": 3.0}

    ignored = score_turn(spec, "I have no idea.", ["search_forum"], timings, tools)
    assert ignored.blame == "generation", "the search found it and the answer skipped it"

    missed = score_turn(spec, "I have no idea.", ["search_forum"], timings, [{"name": "x", "result": {}}])
    assert missed.blame == "retrieval"

    lazy = score_turn(spec, "I have no idea.", [], timings, [])
    assert lazy.blame == "no lookup"

    good = score_turn(spec, "That is a vacuum leak.", ["search_forum"], timings, tools)
    assert good.blame is None and good.passed
