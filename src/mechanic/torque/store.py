"""SQLite storage for Torque sensor uploads.

A "device" is the Torque ``id`` parameter (one phone / one simulated car).
"""

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from mechanic.torque.pids import PIDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    device TEXT NOT NULL,
    ts_ms INTEGER NOT NULL,
    pid INTEGER NOT NULL,
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS readings_device_pid_ts ON readings (device, pid, ts_ms);

CREATE TABLE IF NOT EXISTS sensor_meta (
    device TEXT NOT NULL,
    pid INTEGER NOT NULL,
    full_name TEXT,
    short_name TEXT,
    unit TEXT,
    PRIMARY KEY (device, pid)
);

CREATE TABLE IF NOT EXISTS dtcs (
    device TEXT NOT NULL,
    code TEXT NOT NULL,
    first_seen_ms INTEGER NOT NULL,
    PRIMARY KEY (device, code)
);
"""

RETENTION_MS = 30 * 60 * 1000


@dataclass
class Reading:
    pid: int
    name: str
    unit: str
    value: float
    ts_ms: int


@dataclass
class Trend:
    pid: int
    name: str
    unit: str
    window_s: int
    samples: int
    first: float
    last: float
    min: float
    max: float
    avg: float
    # Change per minute from a least-squares fit over the window.
    slope_per_min: float


class TorqueStore:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def add_upload(
        self,
        device: str,
        ts_ms: int,
        values: dict[int, float],
        meta: dict[int, dict[str, str]] | None = None,
    ) -> None:
        with self._lock, self._db:
            if values:
                self._db.executemany(
                    "INSERT INTO readings (device, ts_ms, pid, value) VALUES (?, ?, ?, ?)",
                    [(device, ts_ms, pid, value) for pid, value in values.items()],
                )
            for pid, m in (meta or {}).items():
                self._db.execute(
                    """INSERT INTO sensor_meta (device, pid, full_name, short_name, unit)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (device, pid) DO UPDATE SET
                         full_name = COALESCE(excluded.full_name, full_name),
                         short_name = COALESCE(excluded.short_name, short_name),
                         unit = COALESCE(excluded.unit, unit)""",
                    (device, pid, m.get("full_name"), m.get("short_name"), m.get("unit")),
                )
            self._db.execute("DELETE FROM readings WHERE device = ? AND ts_ms < ?", (device, ts_ms - RETENTION_MS))

    def _describe(self, device: str, pid: int) -> tuple[str, str]:
        row = self._db.execute(
            "SELECT full_name, unit FROM sensor_meta WHERE device = ? AND pid = ?", (device, pid)
        ).fetchone()
        known = PIDS.get(pid)
        name = (row and row[0]) or (known and known.full_name) or f"PID 0x{pid:x}"
        unit = (row and row[1]) or (known and known.unit) or ""
        return name, unit

    def latest(self, device: str, max_age_s: float | None = None) -> list[Reading]:
        with self._lock:
            rows = self._db.execute(
                """SELECT r.pid, r.value, r.ts_ms FROM readings r
                   JOIN (SELECT pid, MAX(ts_ms) AS ts_ms FROM readings WHERE device = ? GROUP BY pid) m
                     ON r.pid = m.pid AND r.ts_ms = m.ts_ms
                   WHERE r.device = ? ORDER BY r.pid""",
                (device, device),
            ).fetchall()
            now_ms = int(time.time() * 1000)
            out = []
            for pid, value, ts_ms in rows:
                if max_age_s is not None and now_ms - ts_ms > max_age_s * 1000:
                    continue
                name, unit = self._describe(device, pid)
                out.append(Reading(pid, name, unit, value, ts_ms))
            return out

    def trend(self, device: str, pid: int, window_s: int = 300) -> Trend | None:
        with self._lock:
            newest = self._db.execute(
                "SELECT MAX(ts_ms) FROM readings WHERE device = ? AND pid = ?", (device, pid)
            ).fetchone()[0]
            if newest is None:
                return None
            rows = self._db.execute(
                "SELECT ts_ms, value FROM readings WHERE device = ? AND pid = ? AND ts_ms >= ? ORDER BY ts_ms",
                (device, pid, newest - window_s * 1000),
            ).fetchall()
            name, unit = self._describe(device, pid)
        values = [v for _, v in rows]
        n = len(rows)
        slope = 0.0
        if n >= 2:
            ts = [t / 60000 for t, _ in rows]
            mt, mv = sum(ts) / n, sum(values) / n
            denom = sum((t - mt) ** 2 for t in ts)
            if denom > 0:
                slope = sum((t - mt) * (v - mv) for t, v in zip(ts, values, strict=True)) / denom
        return Trend(
            pid=pid,
            name=name,
            unit=unit,
            window_s=window_s,
            samples=n,
            first=values[0],
            last=values[-1],
            min=min(values),
            max=max(values),
            avg=sum(values) / n,
            slope_per_min=slope,
        )

    def set_dtcs(self, device: str, codes: list[str]) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._db:
            if codes:
                placeholders = ",".join("?" * len(codes))
                self._db.execute(
                    f"DELETE FROM dtcs WHERE device = ? AND code NOT IN ({placeholders})", (device, *codes)
                )
            else:
                self._db.execute("DELETE FROM dtcs WHERE device = ?", (device,))
            self._db.executemany(
                "INSERT OR IGNORE INTO dtcs (device, code, first_seen_ms) VALUES (?, ?, ?)",
                [(device, c, now_ms) for c in codes],
            )

    def get_dtcs(self, device: str) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT code FROM dtcs WHERE device = ? ORDER BY first_seen_ms, code", (device,)
            ).fetchall()
        return [r[0] for r in rows]

    def clear_device(self, device: str) -> None:
        with self._lock, self._db:
            for table in ("readings", "sensor_meta", "dtcs"):
                self._db.execute(f"DELETE FROM {table} WHERE device = ?", (device,))
