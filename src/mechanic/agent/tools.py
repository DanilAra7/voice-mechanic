"""The agent's tools: live car data, trouble codes, and three knowledge sources.

Tool results are written to be *spoken*: short, already interpreted where interpretation is
cheap (normal ranges alongside readings), and truncated so the model isn't tempted to read a
forum post aloud verbatim.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from mechanic.knowledge.dtc import DtcDatabase
from mechanic.knowledge.search import SearchIndex
from mechanic.torque.pids import BY_SHORT_NAME, PIDS
from mechanic.torque.store import TorqueStore
from mechanic.vehicles import VEHICLES, Vehicle

MAX_HIT_CHARS = 600
# Long enough to show a direction, short enough to be about now rather than about the drive.
TREND_WINDOW_S = 180
SENSOR_ALIASES = {
    "coolant": 0x05,
    "coolant temp": 0x05,
    "engine temp": 0x05,
    "temperature": 0x05,
    "oil": 0x5C,
    "oil temp": 0x5C,
    "rpm": 0x0C,
    "revs": 0x0C,
    "engine speed": 0x0C,
    "speed": 0x0D,
    "load": 0x04,
    "throttle": 0x11,
    "maf": 0x10,
    "airflow": 0x10,
    "map": 0x0B,
    "boost": 0x0B,
    "intake temp": 0x0F,
    "iat": 0x0F,
    "timing": 0x0E,
    "fuel level": 0x2F,
    "fuel": 0x2F,
    "voltage": 0x42,
    "volts": 0x42,
    "battery": 0x42,
    "short term fuel trim": 0x06,
    "stft": 0x06,
    "short fuel trim": 0x06,
    "long term fuel trim": 0x07,
    "ltft": 0x07,
    "long fuel trim": 0x07,
    "fuel trim": 0x06,
    "ambient": 0x46,
}


@dataclass(frozen=True)
class Band:
    """What counts as normal for one sensor, and what the numbers outside it mean."""

    low: float
    high: float
    below: str = "a little low"
    above: str = "a little high"
    # Past these it is not "a bit off" any more, and the wording stops being gentle.
    far_below: float | None = None
    far_above: float | None = None
    well_below: str = "TOO LOW"
    well_above: str = "TOO HIGH"


def band(pid: int, vehicle: Vehicle | None) -> Band | None:
    """The thresholds, as numbers rather than as a sentence to be interpreted.

    Only the sensors whose reading means something on its own. Engine load and RPM depend on
    what the car is doing at that instant, so there is no honest verdict to give for them.
    """
    match pid:
        case 0x05:
            target = vehicle.coolant_target_c if vehicle else 90
            return Band(
                target - 10,
                target + 10,
                below="below operating temperature",
                above="a little hotter than normal",
                far_above=110,
                well_above="OVERHEATING",
            )
        case 0x5C:
            return Band(80, 110, below="below operating temperature", above="a little hot")
        case 0x06 | 0x07:
            return Band(
                -10,
                10,
                below="slightly rich",
                above="slightly lean",
                far_below=-15,
                far_above=15,
                well_below="TOO RICH",
                well_above="TOO LEAN",
            )
        case 0x42:
            return Band(
                13.5,
                14.7,
                below="low",
                above="high",
                far_below=13.0,
                well_below="NOT CHARGING",
            )
    return None


def verdict(pid: int, value: float, vehicle: Vehicle | None) -> str | None:
    """Whether this reading is normal — decided here, not left to the model to work out.

    Measured 2026-09-20: handed "94.9 °C" and "normal 85-105 °C", the agent told the driver the
    engine was overheating. The comparison is arithmetic and belongs in code, the same way the
    safety rule does; a sentence the model is expected to apply is a sentence it can skip.
    """
    b = band(pid, vehicle)
    if b is None:
        return None
    if b.far_above is not None and value > b.far_above:
        return b.well_above
    if b.far_below is not None and value < b.far_below:
        return b.well_below
    if value > b.high:
        return b.above
    if value < b.low:
        return b.below
    return "normal"


def normal_range(pid: int, vehicle: Vehicle | None) -> str:
    """One-line expectation for a healthy engine, so the model can judge a reading."""
    match pid:
        case 0x05:
            target = vehicle.coolant_target_c if vehicle else 90
            return f"normal {target - 10:.0f}-{target + 10:.0f} °C once warm; over 110 °C is overheating"
        case 0x5C:
            return "normal 80-110 °C once warm"
        case 0x0C:
            idle = vehicle.idle_rpm if vehicle else 700
            return f"idle around {idle} rpm, steady within ±50"
        case 0x06 | 0x07:
            return "normal within ±10 %; beyond +15 % means lean, below -15 % rich"
        case 0x42:
            return "engine running: 13.5-14.7 V; below 13 V means the alternator isn't charging"
        case 0x04:
            return "idle 15-30 %, cruise 25-50 %, hard acceleration 70-100 %"
        case 0x0F:
            return "ambient +5-15 °C, higher when idling after a drive"
        case _:
            return ""


def describe_trend(trend) -> str:
    """Which way a reading is going, in words, so nobody has to subtract two numbers."""
    minutes = round(trend.window_s / 60)
    if abs(trend.slope_per_min) < 0.05:
        return f"steady over the last {minutes} minutes"
    direction = "rising" if trend.slope_per_min > 0 else "falling"
    return f"{direction} {abs(trend.slope_per_min):.1f} {trend.unit} per minute over the last {minutes} minutes"


@dataclass
class Session:
    """Per-conversation state the tools read and write."""

    device: str
    vehicle_id: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def vehicle(self) -> Vehicle | None:
        return VEHICLES.get(self.vehicle_id) if self.vehicle_id else None


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_vehicle",
            "description": "Get the car currently selected for this conversation (make, model, generation, engine). Call this first if you don't know which car the driver has.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_vehicle",
            "description": "Set which car the driver has, when they tell you. Returns the matched car.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What the driver said, e.g. '2012 Audi A4' or 'Honda Civic 2018'.",
                    }
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_live_data",
            "description": "Read the car's current sensor values and any active trouble codes from the OBD-II adapter. Use this whenever the driver describes a symptom happening now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sensors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of sensors, e.g. ['coolant', 'fuel trim', 'voltage']. Omit to get all of them.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sensor_trend",
            "description": "See how one sensor changed over the last few minutes: first/last value, min, max, and rate of change per minute. Use it to tell a rising temperature from a steady one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sensor": {"type": "string", "description": "Sensor name, e.g. 'coolant', 'voltage', 'rpm'."},
                    "minutes": {"type": "number", "description": "Window length in minutes (default 5)."},
                },
                "required": ["sensor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_dtc",
            "description": "Look up what an OBD-II trouble code means, its likely causes ranked by likelihood, symptoms and repair difficulty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code such as 'P0171'. Spoken forms like 'P zero one seven one' work too.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_how_to",
            "description": "Find the step-by-step instructions for doing a job on this car: checking or changing fluids, filters, bulbs, wipers, battery, fuses, where something is and how to reach it. Use this for any 'how do I...', 'how hard is it to...', 'walk me through...' or 'where do I find...' question. Never answer those from memory — the steps differ between cars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What the driver wants to do, e.g. 'check coolant level'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_owner_reports",
            "description": "Find what owners of THIS model reported: whether a fault is a known weak point on this car, how common it is, how they fixed it. Use this whenever the question is about this model specifically — 'is this common on my car', 'do other owners get this', 'known problem on this engine'. For how a fault works in general, use search_forum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The symptom, e.g. 'coolant leak behind engine'."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_forum",
            "description": "Ask what experienced mechanics say about a symptom in general: why it happens, how to tell two causes apart, how someone would track it down. Use this for any 'why would...', 'what causes...', 'how do mechanics...' or 'how would someone find...' question. This is about the fault, not about this particular model — for that use search_owner_reports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The diagnostic question, e.g. 'high fuel trim at idle only'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


class ToolRunner:
    def __init__(self, store: TorqueStore, dtc: DtcDatabase, index: SearchIndex | None = None):
        self.store = store
        self.dtc = dtc
        self.index = index

    def call(self, name: str, args: dict[str, Any], session: Session) -> dict[str, Any]:
        try:
            handler = getattr(self, f"_{name}")
        except AttributeError:
            return {"error": f"Unknown tool '{name}'. Available: {', '.join(TOOL_NAMES)}"}
        try:
            return handler(session, **(args or {}))
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}

    # --- vehicle ---------------------------------------------------------

    def _get_vehicle(self, session: Session) -> dict:
        v = session.vehicle
        if not v:
            return {
                "vehicle": None,
                "message": "No car selected yet. Ask the driver for year, make and model.",
                "known_vehicles": [x.title for x in VEHICLES.values()],
            }
        return {"vehicle": v.title, "engine": v.engine, "years": list(v.years), "generation": v.generation}

    def _set_vehicle(self, session: Session, description: str) -> dict:
        text = description.lower()
        matches = [
            v
            for v in VEHICLES.values()
            if v.make.lower() in text and v.model.lower().replace("-", "") in text.replace("-", "")
        ]
        if not matches:
            return {
                "error": f"'{description}' is not one of the supported cars.",
                "known_vehicles": [v.title for v in VEHICLES.values()],
            }
        years = [int(y) for y in re.findall(r"\b(19[89]\d|20[0-3]\d)\b", text)]
        chosen = next((v for v in matches if any(v.covers_year(y) for y in years)), matches[0])
        session.vehicle_id = chosen.id
        note = ""
        if years and not any(chosen.covers_year(y) for y in years):
            note = f" Note: only the {chosen.years[0]}-{chosen.years[1]} generation is in the knowledge base."
        return {"vehicle": chosen.title, "engine": chosen.engine, "message": f"Selected {chosen.title}.{note}"}

    # --- live data -------------------------------------------------------

    def _resolve_sensor(self, name: str) -> int | None:
        key = name.strip().lower()
        if key in SENSOR_ALIASES:
            return SENSOR_ALIASES[key]
        if key in BY_SHORT_NAME:
            return BY_SHORT_NAME[key].pid
        return next((pid for pid, p in PIDS.items() if key in p.full_name.lower()), None)

    def _read_live_data(self, session: Session, sensors: list[str] | None = None) -> dict:
        readings = self.store.latest(session.device, max_age_s=30)
        if not readings:
            return {
                "connected": False,
                "message": "No live data — the OBD-II adapter isn't sending anything right now.",
            }
        wanted = {self._resolve_sensor(s) for s in sensors} - {None} if sensors else None
        vehicle = session.vehicle
        out, abnormal = [], []
        for r in readings:
            if wanted and r.pid not in wanted:
                continue
            item = {"sensor": r.name, "value": round(r.value, 2), "unit": r.unit}
            if expected := normal_range(r.pid, vehicle):
                item["expected"] = expected
            if (status := verdict(r.pid, r.value, vehicle)) is not None:
                item["status"] = status
                if status != "normal":
                    abnormal.append(f"{r.name} {r.value:.1f} {r.unit} is {status}")
                # Drivers describe a direction — "it keeps climbing" — and the agent used to
                # agree with them out of politeness. Answer it from the log instead.
                if trend := self.store.trend(session.device, r.pid, window_s=TREND_WINDOW_S):
                    item["trend"] = describe_trend(trend)
            out.append(item)
        codes = self.store.get_dtcs(session.device)
        return {
            "connected": True,
            "readings": out,
            "trouble_codes": codes,
            "verdict": (
                "Out of range: " + "; ".join(abnormal)
                if abnormal
                else "Every reading that can be judged is normal for this car."
            ),
            "message": "No trouble codes stored." if not codes else f"Active trouble codes: {', '.join(codes)}",
        }

    def _get_sensor_trend(self, session: Session, sensor: str, minutes: float = 5) -> dict:
        pid = self._resolve_sensor(sensor)
        if pid is None:
            return {
                "error": f"Unknown sensor '{sensor}'.",
                "known_sensors": sorted({p.short_name for p in PIDS.values()}),
            }
        trend = self.store.trend(session.device, pid, window_s=int(minutes * 60))
        if not trend or trend.samples < 2:
            return {"sensor": sensor, "message": "Not enough data yet — the adapter has only just started sending."}
        direction = "rising" if trend.slope_per_min > 0.05 else "falling" if trend.slope_per_min < -0.05 else "steady"
        return {
            "sensor": trend.name,
            "unit": trend.unit,
            "window_minutes": round(trend.window_s / 60, 1),
            "first": round(trend.first, 2),
            "last": round(trend.last, 2),
            "min": round(trend.min, 2),
            "max": round(trend.max, 2),
            "average": round(trend.avg, 2),
            "change_per_minute": round(trend.slope_per_min, 2),
            "direction": direction,
            "expected": normal_range(pid, session.vehicle),
        }

    # --- knowledge -------------------------------------------------------

    def _lookup_dtc(self, session: Session, code: str) -> dict:
        found = self.dtc.lookup(code)
        if not found:
            return {"error": f"'{code}' is not a valid OBD-II code. Codes look like P0171, C0035, U0100."}
        return {
            "code": found.code,
            "title": found.title,
            "summary": found.summary(),
            "related_codes": found.related_codes[:5],
        }

    def _search(self, session: Session, query: str, source: str, limit: int = 3) -> dict:
        if self.index is None:
            return {"error": "The knowledge index isn't loaded on this server."}
        hits = self.index.search(query, sources=(source,), vehicle_id=session.vehicle_id, limit=limit)
        return {
            "query": query,
            "vehicle": session.vehicle.title if session.vehicle else None,
            "results": [
                {
                    "title": h.title,
                    "text": h.text[:MAX_HIT_CHARS],
                    "url": h.url,
                    "vehicle_year": h.year,
                    "solved": h.solved,
                }
                for h in hits
            ],
        }

    def _search_how_to(self, session: Session, query: str) -> dict:
        return self._search(session, query, "carcarekiosk")

    def _search_owner_reports(self, session: Session, query: str) -> dict:
        return self._search(session, query, "startmycar")

    def _search_forum(self, session: Session, query: str) -> dict:
        return self._search(session, query, "stackexchange")
