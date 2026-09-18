"""WebSocket transport for the spoken conversation.

Only moves bytes and events: the microphone arrives as 16 kHz mono PCM, replies leave as
24 kHz mono PCM, and everything about the conversation itself lives in `VoiceSession`.

WebSocket rather than WebRTC because the demo is published through a Cloudflare Tunnel, which
does not carry UDP.

    client -> server   binary: int16 little-endian PCM at 16 kHz
                       json:   {"type": "hello", "device": ..., "vehicle": ...} | {"type": "reset"}
    server -> client   binary: int16 little-endian PCM at the rate given in "ready"
                       json:   ready | transcript | sentence | tool | audio_start | turn_end | flush | error
"""

import logging
import os

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mechanic.agent.loop import AgentLoop
from mechanic.agent.tools import Session, ToolRunner
from mechanic.knowledge.dtc import DtcDatabase
from mechanic.knowledge.search import INDEX_DIR, SearchIndex
from mechanic.torque.store import TorqueStore
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

            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (payload := message.get("bytes")) is not None:
                    await session.push_audio(pcm_to_float(payload))
                elif (text := message.get("text")) is not None:
                    import json

                    if json.loads(text).get("type") == "reset":
                        agent.reset()
                        session.detector.reset()
                        await ws.send_json({"type": "reset_done"})
        except WebSocketDisconnect:
            pass
        except Exception as e:  # a dead socket must not take the server with it
            log.exception("voice socket failed")
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    return router
