"""WebSocket transport for the spoken conversation.

Only moves bytes and events: the microphone arrives as 16 kHz mono PCM, replies leave as
24 kHz mono PCM, and everything about the conversation itself lives in `VoiceSession`.

WebSocket rather than WebRTC because the demo is published through a Cloudflare Tunnel, which
does not carry UDP.

    client -> server   binary: int16 little-endian PCM at 16 kHz
                       json:   {"type": "hello", "device": ..., "vehicle": ...}
                               | {"type": "reset"} | {"type": "vehicle", "vehicle": ...}
                               | {"type": "start_of_speech"} | {"type": "end_of_speech"}
                               | {"type": "ping", "t": ...}
    server -> client   binary: int16 little-endian PCM at the rate given in "ready"
                       json:   ready | transcript | sentence | tool | audio_start | turn_end | flush | error
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mechanic.agent.loop import FIXED_LINES, AgentLoop
from mechanic.agent.tools import Session, ToolRunner
from mechanic.knowledge.dtc import DtcDatabase
from mechanic.knowledge.search import INDEX_DIR, SearchIndex
from mechanic.torque.store import TorqueStore
from mechanic.vehicles import VEHICLES
from mechanic.voice.asr import SAMPLE_RATE as MIC_RATE
from mechanic.voice.asr import Recognizer
from mechanic.voice.session import VoiceSession
from mechanic.voice.tts import Synthesiser
from mechanic.voice.vad import TurnDetector

log = logging.getLogger(__name__)
INT16_SCALE = 32767.0


def pcm_to_float(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / INT16_SCALE


def float_to_pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * INT16_SCALE).astype("<i2").tobytes()


# One JSON line per turn. A file rather than the database because it outlives the process, is
# readable with tail, and nothing about it needs a schema migration at eleven at night.
LATENCY_LOG = Path(os.environ.get("MECHANIC_LATENCY_LOG") or "data/cache/client_latency.jsonl")


def record_client_latency(command: dict, session_id: str, server: object | None = None) -> None:
    try:
        client_ms = float(command.get("client_ms"))
    except (TypeError, ValueError):
        return
    if not 0 < client_ms < 120_000:  # a clock that jumped is not a measurement
        return
    row = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": session_id,
        "client_ms": round(client_ms),
        "rtt_ms": command.get("rtt_ms"),
        "hands_free": bool(command.get("hands_free")),
    }
    # The server's own stages for the same turn. Without them the browser's number is a single
    # figure nobody can act on: it says the wait was long, not which part of it was.
    for stage in ("asr_ms", "first_sentence_ms", "first_audio_ms", "total_ms", "tool_ms"):
        value = getattr(server, stage, None)
        if isinstance(value, int | float):
            row[stage] = round(value)
    if tools := getattr(server, "tools", None):
        row["tools"] = list(tools)
    try:
        LATENCY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with LATENCY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:  # a full disk must not end the conversation
        log.warning("could not record client latency: %s", e)


def latency_summary() -> dict:
    """p50 and p95 over every turn anyone has ever waited through, on this machine."""
    if not LATENCY_LOG.exists():
        return {"turns": 0}
    rows = []
    for line in LATENCY_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    waits = sorted(r["client_ms"] for r in rows if isinstance(r.get("client_ms"), int | float))
    if not waits:
        return {"turns": 0}
    rtts = sorted(r["rtt_ms"] for r in rows if isinstance(r.get("rtt_ms"), int | float))

    def q(xs: list, p: float) -> int:
        return round(xs[min(len(xs) - 1, int(len(xs) * p))])

    def med(key: str) -> int | None:
        xs = sorted(r[key] for r in rows if isinstance(r.get(key), int | float))
        return round(xs[len(xs) // 2]) if xs else None

    asr, sentence, audio = med("asr_ms"), med("first_sentence_ms"), med("first_audio_ms")
    stages = {}
    if asr and sentence and audio:
        stages = {
            "recognition_ms": asr,
            "model_ms": sentence - asr,
            "synthesis_ms": audio - sentence,
            "server_total_ms": audio,
            # What the browser waited beyond anything the server did: the network each way plus
            # the audio pipeline in the tab. Named rather than left as an unexplained remainder.
            "network_and_browser_ms": q(waits, 0.5) - audio,
        }
    return {
        "turns": len(waits),
        **stages,
        "client_p50_ms": q(waits, 0.5),
        "client_p95_ms": q(waits, 0.95),
        "client_min_ms": waits[0],
        "client_max_ms": waits[-1],
        "rtt_p50_ms": q(rtts, 0.5) if rtts else None,
        "hands_free_turns": sum(1 for r in rows if r.get("hands_free")),
    }


class SharedModels:
    """Recognition and synthesis are loaded once for the process; the GPU has room for one copy."""

    def __init__(self):
        self.recognizer = Recognizer()
        self.synthesiser = Synthesiser()
        self.dtc = DtcDatabase()
        self.index = SearchIndex() if (INDEX_DIR / "dense.npy").exists() else None
        if self.index is None:
            log.warning("No dense index built - the search tools will be weaker")
        self._warm = False

    async def warm_up(self) -> None:
        """Both models pay seconds on their first call. Spend that before anyone connects."""
        if self._warm:
            return
        import asyncio

        await asyncio.to_thread(self.recognizer.warm_up)
        await asyncio.to_thread(self.synthesiser.warm_up)
        if prime := getattr(self.synthesiser, "prime", None):
            # A few seconds here buy the synthesiser's whole latency back on every turn that
            # starts with one of these lines, which is most of them.
            await asyncio.to_thread(prime, FIXED_LINES)
        if self.index is not None:
            await asyncio.to_thread(self.index.search, "warm up the query embedder", limit=1)
        self._warm = True


def build_router(store: TorqueStore, models: SharedModels) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/voice")
    async def voice(ws: WebSocket) -> None:
        await ws.accept()
        session: VoiceSession | None = None
        try:
            hello = await ws.receive_json()
            device = hello.get("device", "demo")
            vehicle = hello.get("vehicle") or None

            async def on_event(event: str, payload: dict) -> None:
                await ws.send_json({"type": event, **payload})

            async def on_audio(chunk: np.ndarray, _rate: int) -> None:
                await ws.send_bytes(float_to_pcm(chunk))

            agent = AgentLoop(
                ToolRunner(store, models.dtc, models.index),
                Session(device=device, vehicle_id=vehicle),
                model=os.environ.get("MECHANIC_LLM_MODEL", "gpt-oss-20b"),
                base_url=os.environ.get("MECHANIC_LLM_BASE_URL", "http://127.0.0.1:8001/v1"),
            )
            session = VoiceSession(
                agent=agent,
                recognizer=models.recognizer,
                synthesiser=models.synthesiser,
                detector=TurnDetector(),
                on_event=on_event,
                on_audio=on_audio,
            )
            await models.warm_up()
            await ws.send_json(
                {
                    "type": "ready",
                    "mic_rate": MIC_RATE,
                    "audio_rate": session.synthesiser_sample_rate,
                    "vehicle": vehicle,
                    "device": device,
                }
            )

            async def control(command: dict) -> None:
                match command.get("type"):
                    case "start_of_speech":
                        # Button down: hold everything until they let go, however long they pause.
                        session.start_of_speech()
                    case "end_of_speech":
                        # Push-to-talk button released: the turn is over because they said so.
                        await session.end_of_speech()
                    case "client_latency":
                        # What the listener waited, measured on their clock and kept on ours.
                        # Every other latency number on this project is the server timing itself,
                        # which cannot see the network, the tunnel or the browser's audio stack.
                        record_client_latency(command, session_id=device, server=session.last_timings)
                    case "ping":
                        # Echoed straight back so the browser can price the network on its own
                        # clock; the latency panel shows it beside the time the server spent.
                        await ws.send_json({"type": "pong", "t": command.get("t")})
                    case "reset":
                        agent.reset()
                        await session.reset()
                        await ws.send_json({"type": "reset_done"})
                    case "vehicle":
                        # The Garage panel can put the driver in a different car mid-session.
                        # The agent states the current car in its system prefix and rebuilds it
                        # when this changes; the history goes too, because every word of it was
                        # about the other car and would be read as being about this one.
                        chosen = command.get("vehicle") or None
                        if chosen is not None and chosen not in VEHICLES:
                            await ws.send_json({"type": "error", "message": f"Unknown vehicle '{chosen}'"})
                            return
                        agent.session.vehicle_id = chosen
                        agent.reset()
                        await session.reset()
                        await ws.send_json({"type": "vehicle", "vehicle": chosen})

            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (payload := message.get("bytes")) is not None:
                    await session.push_audio(pcm_to_float(payload))
                elif (text := message.get("text")) is not None:
                    await control(json.loads(text))
        except WebSocketDisconnect:
            pass
        except Exception as e:  # a dead socket must not take the server with it
            log.exception("voice socket failed")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    return router
