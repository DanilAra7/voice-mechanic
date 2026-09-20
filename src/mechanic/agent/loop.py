"""The agent loop: stream tokens from an OpenAI-compatible server, run tools, stream again.

Latency-shaped on purpose:
- sentences are emitted as soon as they are complete, so TTS can start before the model is done;
- a filler line ("Let me check that...") is emitted when a slow tool starts, so the driver hears
  something while the search runs;
- every stage is timestamped and reported through `on_event` for the latency panel in the UI.
"""

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from mechanic.agent.prompt import build_system_prompt
from mechanic.agent.safety import DO_NOT_DRIVE, PULL_OVER, safety_warning
from mechanic.agent.tools import TOOL_SCHEMAS, Session, ToolRunner

# Any tool at all is worth a spoken line: what costs seconds is the model round around it,
# not the lookup itself, and silence is what the driver notices.
FILLERS = {
    "search_forum": "Let me check what other mechanics say.",
    "search_owner_reports": "Let me see what other owners report.",
    "search_how_to": "Let me pull up the steps.",
    "read_live_data": "Checking your live data.",
}
# Everything the agent ever says word for word. Synthesised at startup so the first sound of a
# turn costs nothing; see Synthesiser.prime.
FIXED_LINES = (*FILLERS.values(), "Let me check that.", PULL_OVER, DO_NOT_DRIVE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MAX_TOOL_ROUNDS = 4
# Roughly how long the driver has to listen: the synthesiser speaks 13.5 characters a second
# (measured over 12 answers on 2026-09-20), so text length is listening time. Without a cap the
# model answered one question with 36 seconds of speech — a fast first word is worth nothing if
# the answer then runs for most of a minute, and they cannot ask the next question until it
# stops. Told to be brief the model agrees and then is not, so the limit is enforced rather than
# requested. The cut lands between sentences, so what they hear is always a finished thought,
# and the sentence that crosses the line is still spoken in full: about ten seconds in practice.
MAX_SPOKEN_CHARS = 140


def situation(session: Session) -> str:
    """The one fact the agent always needs, stated instead of looked up.

    Measured on day 3: without this the model spends a tool round on get_vehicle, or worse,
    asks the driver which car this is when the session already knows.
    """
    v = session.vehicle
    if not v:
        return "The car has not been identified yet."
    return (
        f"The driver's car is a {v.title}. Live sensor data for it is available. "
        "You already know this, so never ask them which car, year or engine they have."
    )


@dataclass
class Turn:
    """Everything one user utterance produced, for the UI and for evaluation."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Set when the guardrail spoke before the model did; kept separate so latency numbers stay honest.
    safety_line: str | None = None
    first_token_ms: float | None = None
    first_sentence_ms: float | None = None
    total_ms: float | None = None
    rounds: int = 0
    # The answer was cut at a sentence boundary because it had run long enough to listen to.
    truncated: bool = False


class AgentLoop:
    def __init__(
        self,
        tools: ToolRunner,
        session: Session,
        model: str,
        base_url: str = "http://127.0.0.1:8001/v1",
        api_key: str = "not-needed",
        temperature: float = 0.3,
        max_tokens: int = 220,
        extra_system: str | None = None,
    ):
        self.tools = tools
        self.session = session
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra_system = extra_system
        self._vehicle_id = session.vehicle_id
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_content()}]

    def _system_content(self) -> str:
        extra = f"{situation(self.session)}\n{self._extra_system}" if self._extra_system else situation(self.session)
        return build_system_prompt(extra)

    def _refresh_system(self) -> None:
        """Rebuild the prefix only when the car actually changed, so it stays cacheable."""
        if self.session.vehicle_id != self._vehicle_id:
            self._vehicle_id = self.session.vehicle_id
            self.messages[0] = {"role": "system", "content": self._system_content()}

    def reset(self) -> None:
        self.messages = self.messages[:1]

    async def respond(
        self,
        user_text: str,
        on_event: Callable[[str, dict], Awaitable[None]] | None = None,
    ) -> Turn:
        """Run one turn; yields nothing but streams sentences through `on_event`."""
        turn = Turn()
        async for _ in self.stream(user_text, turn=turn, on_event=on_event):
            pass
        return turn

    async def stream(
        self,
        user_text: str,
        turn: Turn | None = None,
        on_event: Callable[[str, dict], Awaitable[None]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield spoken sentences as they become available."""
        turn = turn or Turn()
        started = time.monotonic()
        self._refresh_system()
        self.messages.append({"role": "user", "content": user_text})

        async def emit(event: str, payload: dict) -> None:
            if on_event:
                await on_event(event, payload)

        spoken = False
        said_chars = 0
        if (warning := safety_warning(user_text)) is not None:
            turn.safety_line = warning
            turn.text += warning + " "
            turn.first_sentence_ms = (time.monotonic() - started) * 1000
            spoken = True
            await emit("sentence", {"text": warning, "safety": True})
            yield warning
            self.messages.append(
                {
                    "role": "system",
                    "content": f'You already told the driver: "{warning}" '
                    "Do not repeat it. Continue with what to check and why.",
                }
            )

        for round_index in range(MAX_TOOL_ROUNDS):
            turn.rounds = round_index + 1
            text, tool_calls, sentences, enough = "", [], [], False
            async for kind, value in self._stream_completion(started, turn):
                if kind == "sentence":
                    sentences.append(value)
                    spoken = True
                    said_chars += len(value)
                    turn.text += value + " "
                    await emit("sentence", {"text": value})
                    yield value
                    if said_chars >= MAX_SPOKEN_CHARS:
                        # Whatever the model is still writing, the driver has heard enough for
                        # one turn. They can always ask for more; they cannot ask for less.
                        turn.truncated = True
                        enough = True
                        break
                elif kind == "text":
                    text = value
                elif kind == "tool_calls":
                    tool_calls = value

            if enough:
                self.messages.append({"role": "assistant", "content": text or " ".join(sentences)})
                break

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                break

            self.messages.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            if not spoken:
                filler = FILLERS.get(tool_calls[0]["function"]["name"], "Let me check that.")
                turn.text += filler + " "
                spoken = True
                # The driver hears this, so it is the first sentence as far as latency goes.
                if turn.first_sentence_ms is None:
                    turn.first_sentence_ms = (time.monotonic() - started) * 1000
                await emit("sentence", {"text": filler, "filler": True})
                yield filler

            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                t0 = time.monotonic()
                result = await asyncio.to_thread(self.tools.call, name, args, self.session)
                took_ms = (time.monotonic() - t0) * 1000
                turn.tool_calls.append(
                    {"name": name, "arguments": args, "duration_ms": round(took_ms), "result": result}
                )
                await emit("tool_call", {"name": name, "arguments": args, "duration_ms": round(took_ms)})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "name": name, "content": json.dumps(result)[:4000]}
                )

        turn.total_ms = (time.monotonic() - started) * 1000
        turn.text = turn.text.strip()
        await emit("turn_end", {"total_ms": turn.total_ms, "tool_calls": len(turn.tool_calls)})

    async def _stream_completion(self, started: float, turn: Turn) -> AsyncIterator[tuple[str, Any]]:
        # How hard the model is allowed to think, when the server is willing to be told per
        # request. Set on the server for the demo; overridable here so the cost of thinking
        # harder can be measured against the same scenarios.
        extra = (
            {"chat_template_kwargs": {"reasoning_effort": effort}}
            if (effort := os.environ.get("MECHANIC_REASONING"))
            else None
        )
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            extra_body=extra,
        )
        text = ""
        pending = ""
        calls: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if turn.first_token_ms is None and (delta.content or delta.tool_calls):
                turn.first_token_ms = (time.monotonic() - started) * 1000
            for tc in delta.tool_calls or []:
                empty_call = {
                    "id": tc.id or f"call_{tc.index}",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                slot = calls.setdefault(tc.index, empty_call)
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["function"]["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["function"]["arguments"] += tc.function.arguments
            if delta.content:
                text += delta.content
                pending += delta.content
                # Emit complete sentences so speech starts before generation finishes.
                while (parts := SENTENCE_END.split(pending, maxsplit=1)) and len(parts) == 2:
                    sentence, pending = parts[0].strip(), parts[1]
                    if sentence:
                        if turn.first_sentence_ms is None:
                            turn.first_sentence_ms = (time.monotonic() - started) * 1000
                        yield "sentence", sentence
        if pending.strip():
            if turn.first_sentence_ms is None:
                turn.first_sentence_ms = (time.monotonic() - started) * 1000
            yield "sentence", pending.strip()
        yield "text", text
        if calls:
            yield "tool_calls", [calls[i] for i in sorted(calls)]
