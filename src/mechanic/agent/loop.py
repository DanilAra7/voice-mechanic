"""The agent loop: stream tokens from an OpenAI-compatible server, run tools, stream again.

Latency-shaped on purpose:
- sentences are emitted as soon as they are complete, so TTS can start before the model is done;
- a filler line ("Let me check that...") is emitted when a slow tool starts, so the driver hears
  something while the search runs;
- every stage is timestamped and reported through `on_event` for the latency panel in the UI.
"""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from mechanic.agent.prompt import build_system_prompt
from mechanic.agent.tools import TOOL_SCHEMAS, Session, ToolRunner

# Tools that hit the index are slow enough (tens of ms plus embedding) to warrant a spoken filler.
SLOW_TOOLS = {"search_forum", "search_owner_reports", "search_how_to"}
FILLERS = {
    "search_forum": "Let me check what other mechanics say.",
    "search_owner_reports": "Let me see what other owners report.",
    "search_how_to": "Let me pull up the steps.",
    "read_live_data": "Checking your live data.",
}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MAX_TOOL_ROUNDS = 4


@dataclass
class Turn:
    """Everything one user utterance produced, for the UI and for evaluation."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    first_token_ms: float | None = None
    first_sentence_ms: float | None = None
    total_ms: float | None = None
    rounds: int = 0


class AgentLoop:
    def __init__(
        self,
        tools: ToolRunner,
        session: Session,
        model: str,
        base_url: str = "http://127.0.0.1:8001/v1",
        api_key: str = "not-needed",
        temperature: float = 0.3,
        max_tokens: int = 400,
        extra_system: str | None = None,
    ):
        self.tools = tools
        self.session = session
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(extra_system)}]

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
        self.messages.append({"role": "user", "content": user_text})

        async def emit(event: str, payload: dict) -> None:
            if on_event:
                await on_event(event, payload)

        for round_index in range(MAX_TOOL_ROUNDS):
            turn.rounds = round_index + 1
            text, tool_calls, sentences = "", [], []
            async for kind, value in self._stream_completion(started, turn):
                if kind == "sentence":
                    sentences.append(value)
                    turn.text += value + " "
                    await emit("sentence", {"text": value})
                    yield value
                elif kind == "text":
                    text = value
                elif kind == "tool_calls":
                    tool_calls = value

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": text})
                break

            self.messages.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            if any(c["function"]["name"] in SLOW_TOOLS for c in tool_calls) and not sentences:
                filler = FILLERS.get(tool_calls[0]["function"]["name"], "One moment.")
                turn.text += filler + " "
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
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_SCHEMAS,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
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
