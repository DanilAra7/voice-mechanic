"""HTTP endpoint compatible with Torque Pro's "Upload to web-server" feature.

Torque sends ``GET <url>?v=..&session=..&id=..&eml=..&time=<epoch ms>&k<pid>=<value>...``
and, in some requests, sensor metadata as ``userFullName<pid>``, ``userShortName<pid>``,
``userUnit<pid>`` / ``defaultUnit<pid>``. The server must answer with the body ``OK!``.
Protocol verified against the Home Assistant torque integration and open-torque-viewer.
"""

import re
import time
from collections.abc import Mapping

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from mechanic.torque.store import TorqueStore

VALUE_KEY = re.compile(r"^k([0-9a-fA-F]+)$")
META_KEY = re.compile(r"^(userFullName|userShortName|userUnit|defaultUnit)([0-9a-fA-F]+)$")
META_FIELD = {
    "userFullName": "full_name",
    "userShortName": "short_name",
    "userUnit": "unit",
    "defaultUnit": "unit",
}


def parse_upload(params: Mapping[str, str]) -> tuple[str, int, dict[int, float], dict[int, dict[str, str]]]:
    """Return (device, ts_ms, values, meta) from Torque query parameters."""
    device = params.get("id") or params.get("eml") or "unknown"
    try:
        ts_ms = int(params["time"])
    except (KeyError, ValueError):
        ts_ms = int(time.time() * 1000)

    values: dict[int, float] = {}
    meta: dict[int, dict[str, str]] = {}
    for key, raw in params.items():
        if m := VALUE_KEY.match(key):
            try:
                values[int(m.group(1), 16)] = float(raw)
            except ValueError:
                continue
        elif m := META_KEY.match(key):
            field = META_FIELD[m.group(1)]
            pid_meta = meta.setdefault(int(m.group(2), 16), {})
            # userUnit wins over defaultUnit when both are present.
            if field == "unit" and m.group(1) == "defaultUnit" and "unit" in pid_meta:
                continue
            pid_meta[field] = raw.replace("\\xC2\\xB0", "°")
    return device, ts_ms, values, meta


def build_router(store: TorqueStore) -> APIRouter:
    router = APIRouter()

    @router.get("/torque", response_class=PlainTextResponse)
    async def torque_upload(request: Request) -> str:
        device, ts_ms, values, meta = parse_upload(request.query_params)
        store.add_upload(device, ts_ms, values, meta)
        return "OK!"

    return router
