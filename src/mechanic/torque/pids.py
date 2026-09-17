"""OBD-II PIDs we simulate, keyed the way Torque Pro uploads them.

Torque sends each sensor as a query parameter ``k<pid in hex>`` (lowercase, no
leading zeros), e.g. ``k5`` = coolant temperature (PID 0x05), ``kc`` = RPM.
Values are metric by default.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pid:
    pid: int
    short_name: str
    full_name: str
    unit: str

    @property
    def torque_key(self) -> str:
        return f"k{self.pid:x}"


PIDS: dict[int, Pid] = {
    p.pid: p
    for p in [
        Pid(0x04, "Load", "Engine Load", "%"),
        Pid(0x05, "Coolant", "Engine Coolant Temperature", "°C"),
        Pid(0x06, "STFT1", "Fuel Trim Bank 1 Short Term", "%"),
        Pid(0x07, "LTFT1", "Fuel Trim Bank 1 Long Term", "%"),
        Pid(0x0B, "MAP", "Intake Manifold Pressure", "kPa"),
        Pid(0x0C, "Revs", "Engine RPM", "rpm"),
        Pid(0x0D, "Speed", "Speed (OBD)", "km/h"),
        Pid(0x0E, "Timing", "Timing Advance", "°"),
        Pid(0x0F, "IAT", "Intake Air Temperature", "°C"),
        Pid(0x10, "MAF", "Mass Air Flow Rate", "g/s"),
        Pid(0x11, "Throttle", "Throttle Position (Manifold)", "%"),
        Pid(0x2F, "Fuel", "Fuel Level (From Engine ECU)", "%"),
        Pid(0x42, "Volts", "Voltage (Control Module)", "V"),
        Pid(0x46, "Ambient", "Ambient Air Temp", "°C"),
        Pid(0x5C, "Oil", "Engine Oil Temperature", "°C"),
    ]
}

BY_KEY: dict[str, Pid] = {p.torque_key: p for p in PIDS.values()}
BY_SHORT_NAME: dict[str, Pid] = {p.short_name.lower(): p for p in PIDS.values()}
