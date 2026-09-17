"""Vehicles supported by the MVP.

Each profile holds the numbers the simulator needs (idle RPM, normal operating
temperatures, ...) and the keys used to locate this vehicle in the scraped
knowledge sources.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Vehicle:
    id: str
    make: str
    model: str
    generation: str
    years: tuple[int, int]
    engine: str
    cylinders: int
    displacement_l: float
    turbo: bool
    idle_rpm: int
    redline_rpm: int
    # Coolant temperature the thermostat regulates to once warm, °C.
    coolant_target_c: float
    fuel_tank_l: float
    # carcarekiosk.com generation page, e.g. "Audi/A4_Quattro/2009".
    carcarekiosk_path: str
    # startmycar.com model slug, e.g. "audi/a4".
    startmycar_slug: str
    # mechanics.stackexchange.com tags that identify this vehicle.
    stackexchange_tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def title(self) -> str:
        return f"{self.years[0]}–{self.years[1]} {self.make} {self.model} ({self.generation}, {self.engine})"

    def covers_year(self, year: int) -> bool:
        return self.years[0] <= year <= self.years[1]


VEHICLES: dict[str, Vehicle] = {
    v.id: v
    for v in [
        Vehicle(
            id="audi_a4_b8",
            make="Audi",
            model="A4",
            generation="B8",
            years=(2009, 2016),
            engine="2.0L TFSI I4 turbo",
            cylinders=4,
            displacement_l=2.0,
            turbo=True,
            idle_rpm=720,
            redline_rpm=6500,
            coolant_target_c=95.0,
            fuel_tank_l=64.0,
            carcarekiosk_path="Audi/A4_Quattro/2009",
            startmycar_slug="audi/a4",
            stackexchange_tags=("audi", "a4"),
        ),
        Vehicle(
            id="honda_accord_9",
            make="Honda",
            model="Accord",
            generation="9th gen",
            years=(2013, 2017),
            engine="2.4L I4",
            cylinders=4,
            displacement_l=2.4,
            turbo=False,
            idle_rpm=700,
            redline_rpm=6800,
            coolant_target_c=90.0,
            fuel_tank_l=65.0,
            carcarekiosk_path="Honda/Accord/2013",
            startmycar_slug="honda/accord",
            stackexchange_tags=("honda", "accord"),
        ),
        Vehicle(
            id="ford_f150_13",
            make="Ford",
            model="F-150",
            generation="13th gen",
            years=(2015, 2020),
            engine="2.7L EcoBoost V6 turbo",
            cylinders=6,
            displacement_l=2.7,
            turbo=True,
            idle_rpm=650,
            redline_rpm=6000,
            coolant_target_c=93.0,
            fuel_tank_l=87.0,
            carcarekiosk_path="Ford/F-150/2015",
            startmycar_slug="ford/f-150",
            stackexchange_tags=("ford", "f-150"),
        ),
        Vehicle(
            id="honda_civic_10",
            make="Honda",
            model="Civic",
            generation="10th gen",
            years=(2016, 2021),
            engine="2.0L I4",
            cylinders=4,
            displacement_l=2.0,
            turbo=False,
            idle_rpm=680,
            redline_rpm=6700,
            coolant_target_c=90.0,
            fuel_tank_l=47.0,
            carcarekiosk_path="Honda/Civic/2016",
            startmycar_slug="honda/civic",
            stackexchange_tags=("honda", "civic"),
        ),
        Vehicle(
            id="toyota_corolla_11",
            make="Toyota",
            model="Corolla",
            generation="11th gen",
            years=(2014, 2019),
            engine="1.8L I4",
            cylinders=4,
            displacement_l=1.8,
            turbo=False,
            idle_rpm=650,
            redline_rpm=6200,
            coolant_target_c=88.0,
            fuel_tank_l=50.0,
            carcarekiosk_path="Toyota/Corolla/2014",
            startmycar_slug="toyota/corolla",
            stackexchange_tags=("toyota", "corolla"),
        ),
    ]
}


def get_vehicle(vehicle_id: str) -> Vehicle:
    try:
        return VEHICLES[vehicle_id]
    except KeyError:
        raise KeyError(f"Unknown vehicle '{vehicle_id}'. Known: {', '.join(VEHICLES)}") from None
