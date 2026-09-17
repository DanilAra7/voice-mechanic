"""The loop is tested against a scripted fake LLM: no server, no model, deterministic."""

import json
import time
from types import SimpleNamespace

import pytest
from conftest import FakeIndex

from mechanic.agent.loop import AgentLoop
from mechanic.agent.tools import Session, ToolRunner
from mechanic.knowledge.dtc import DtcDatabase
from mechanic.torque.store import TorqueStore


def content_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))])


def tool_chunk(index, name=None, args=None, call_id=None):
    fn = SimpleNamespace(name=name, arguments=args)
    tc = SimpleNamespace(index=index, id=call_id, function=fn)
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))])


class FakeLLM:
    """Replays scripted responses; records the messages it was asked to complete."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        # Snapshot the message list: the loop keeps mutating the same object.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        chunks = self.scripts.pop(0)

        async def gen():
            for c in chunks:
                yield c

        return gen()


def build_loop(scripts, vehicle_id="audi_a4_b8"):
    store = TorqueStore()
    store.add_upload("dev1", int(time.time() * 1000), {0x05: 118.0, 0x42: 14.1})
    store.set_dtcs("dev1", ["P0217"])
    loop = AgentLoop(
        ToolRunner(store, DtcDatabase(), FakeIndex()),
        Session(device="dev1", vehicle_id=vehicle_id),
        model="fake",
    )
    loop.client = FakeLLM(scripts)
    return loop


async def test_plain_answer_streams_sentences_early():
    loop = build_loop([[content_chunk("Sure thing. "), content_chunk("That sounds normal. "), content_chunk("Bye")]])
    events = []

    async def record(event, payload):
        events.append((event, payload))

    sentences = [s async for s in loop.stream("is 90 degrees ok?", on_event=record)]
    assert sentences == ["Sure thing.", "That sounds normal.", "Bye"]
    assert [e for e, _ in events] == ["sentence", "sentence", "sentence", "turn_end"]


async def test_tool_call_round_trip_and_filler():
    scripts = [
        [
            tool_chunk(0, name="read_live_", args="", call_id="c1"),
            tool_chunk(0, name="data", args='{"sensors":'),
            tool_chunk(0, args=' ["coolant"]}'),
        ],
        [content_chunk("Your coolant is at one eighteen. "), content_chunk("Pull over now.")],
    ]
    loop = build_loop(scripts)
    turn = await loop.respond("my temperature gauge is climbing")

    assert [c["name"] for c in turn.tool_calls] == ["read_live_data"]
    assert turn.tool_calls[0]["arguments"] == {"sensors": ["coolant"]}
    assert turn.tool_calls[0]["result"]["connected"] is True
    assert turn.rounds == 2
    assert turn.first_token_ms is not None and turn.total_ms > 0
    assert "Pull over now." in turn.text

    # The tool result went back to the model as a `tool` message tied to the call id.
    second_request = loop.client.requests[1]["messages"]
    tool_message = second_request[-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "c1"
    assert "Coolant" in json.loads(tool_message["content"])["readings"][0]["sensor"]


async def test_slow_tool_emits_filler_before_the_search():
    scripts = [
        [tool_chunk(0, name="search_forum", args='{"query": "overheating at idle"}', call_id="c9")],
        [content_chunk("Most likely a failing water pump.")],
    ]
    loop = build_loop(scripts)
    spoken = [s async for s in loop.stream("what could it be?")]
    assert spoken[0] == "Let me check what other mechanics say."
    assert spoken[-1] == "Most likely a failing water pump."


async def test_malformed_tool_arguments_do_not_crash():
    scripts = [
        [tool_chunk(0, name="lookup_dtc", args="{not json", call_id="c2")],
        [content_chunk("I could not read that code.")],
    ]
    loop = build_loop(scripts)
    turn = await loop.respond("what is P0217?")
    assert turn.tool_calls[0]["arguments"] == {}
    assert "error" in turn.tool_calls[0]["result"]


async def test_tool_rounds_are_capped():
    call = [tool_chunk(0, name="read_live_data", args="{}", call_id="c3")]
    loop = build_loop([list(call) for _ in range(6)])
    turn = await loop.respond("check again")
    assert turn.rounds == 4
    assert len(turn.tool_calls) == 4


@pytest.mark.parametrize("vehicle", ["audi_a4_b8", None])
async def test_system_prompt_is_first_and_stable(vehicle):
    loop = build_loop([[content_chunk("Hi.")]], vehicle_id=vehicle)
    await loop.respond("hello")
    messages = loop.client.requests[0]["messages"]
    assert messages[0]["role"] == "system" and "Dex" in messages[0]["content"]
    loop.reset()
    assert len(loop.messages) == 1
