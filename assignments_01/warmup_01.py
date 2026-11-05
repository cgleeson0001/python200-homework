"""Week 1 warmups: classes, dataclasses, Pydantic and pytest."""
from dataclasses import dataclass, field, FrozenInstanceError
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Self
import pytest

# --- Classes ---
# Q1
class Thermometer:
    """Store Celsius readings for one location."""
    def __init__(self, location: str, readings: list[float] | None = None) -> None:
        self.location = location
        self.readings = list(readings) if readings is not None else []

    def add(self, reading: float) -> None:
        self.readings.append(reading)

    def average(self) -> float | None:
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self) -> float | None:
        return max(self.readings) if self.readings else None

    # Q2
    def __repr__(self) -> str:
        return f"Thermometer(location={self.location!r}, n_readings={len(self.readings)}, average={self.average()})"

# Q1: Empty readings need a guard because len=0 would cause ZeroDivisionError.
# Q2: Without __repr__, print displays an object type and memory address,
# which does not reveal the useful attributes when debugging.

# Q3
class TemperatureAlert:
    """Detect readings above a reusable Celsius alert threshold."""
    def __init__(self, threshold: float = 30.0) -> None:
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        """Return readings strictly above this alert's threshold."""
        return [r for r in thermometer.readings if r > self.threshold]

# Q3: Keeping threshold on each alert lets the same alert configuration be
# reused across 20 thermometers without supplying the threshold 20 times.

# --- Dataclasses, Type Hints, and Docstrings ---
# Q1 and Q2
@dataclass(frozen=True)
class Station:
    """A physical weather station with its location and elevation in meters."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

# Q1: Dataclasses compare all field values by default, while the original
# ordinary class used identity comparison (so two distinct instances !=).
# Q2: frozen=True also allows hashing when fields are hashable, so Station
# objects can be set elements and dictionary keys.

# Q3
@dataclass
class StationBatch:
    """Stations located within one named region."""
    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Append a station to this region's batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the highest-elevation station, if any."""
        return max(self.stations, key=lambda station: station.elevation, default=None)

# Q3: Trying `stations: list[Station] = []` raises:
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
# Mutable default lists could otherwise be shared across instances.

# --- Pydantic ---
# Q1 and Q4
class Reading(BaseModel):
    """Validate one incoming station sensor reading."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def reject_failed_sensor(self) -> Self:
        """Reject unlikely readings indicating a failed sensor."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("sensor failure: zero humidity below -40 C")
        return self

# Q2: Pydantic can coerce a numeric string like '21.5' to float, but cannot
# turn a nonnumeric string like 'very humid' into a float.
# Q3: Reporting all errors together makes fixing invalid inputs more efficient.
# Q4: A model validator is needed because the rule relates two fields;
# individual Field constraints cannot compare humidity and temperature.

# --- pytest ---
# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature into Fahrenheit."""
    return celsius * 9 / 5 + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)
    # approx handles floating-point rounding in computed decimal values.

# Q2
def mean(values: list[float]) -> float:
    """Return the arithmetic mean or reject empty input."""
    if not values:
        raise ValueError("values list cannot be empty")
    return sum(values) / len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])
    # match also checks the error message; raises alone checks only its type.

# Q3
@pytest.mark.parametrize("values,expected", [([4], 4), ([1, 3], 2), ([-3, -1], -2), ([0, 5, 10], 5)])
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)
# Parametrization avoids repeated test functions and reports each case.
# Pytest results: 6 passed in 1.09s

# Q4: Temporarily change 9 / 5 above to 9 / 4 and run pytest warmup_01.py -v.
# ================== FAILURES ==================
#_________ test_celsius_to_fahrenheit _________

#    def test_celsius_to_fahrenheit():
#        assert celsius_to_fahrenheit(0) == 32
#>       assert celsius_to_fahrenheit(100) == 212
#E       assert 257.0 == 212
#E        +  where 257.0 = celsius_to_fahrenheit(100)
#
#warmup_01.py:112: AssertionError
#========== short test summary info ===========
#FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# Pytest's actual and expected values make it clear how the formula differs.

def demonstrate() -> None:
    """Display results required by the warmup instructions."""
    print("--- Classes Q1-Q3 ---")
    thermometer = Thermometer("Louisville")
    for temperature in [25, 30, 22, 28]:
        thermometer.add(temperature)
    print("Average:", thermometer.average())
    print("Hottest:", thermometer.hottest())
    print(thermometer)
    print([thermometer, Thermometer("Charlotte", [18, 19])])
    print("Alert over 26:", TemperatureAlert(26).breaches(thermometer))
    print("Alert over 29:", TemperatureAlert(29).breaches(thermometer))
    print("--- Dataclasses Q1-Q3 ---")
    station_a = Station("ABC", "Hill", 35.0, -80.0, 200.0)
    station_b = Station("ABC", "Hill", 35.0, -80.0, 200.0)
    print("Equal stations:", station_a == station_b)
    try:
        station_a.name = "Updated"
    except FrozenInstanceError as error:
        print("Frozen error:", error)
    station_c = Station("DEF", "Peak", 36.0, -81.0, 350.0)
    print("Number of unique stations:", len({station_a, station_b, station_c}))
    batch = StationBatch("Southeast")
    batch.add(station_a)
    batch.add(station_c)
    print("Highest station:", batch.highest())
    print("--- Pydantic Q1-Q4 ---")
    valid = dict(station_id="ABC", timestamp="2026-04-08T00:00", temperature_c=21.5, humidity=40)
    print("Valid:", Reading(**valid))
    for change in [dict(timestamp=None), dict(temperature_c=150.0), dict(humidity="very humid")]:
        invalid = valid.copy()
        invalid.update(change)
        if change == {"timestamp": None}:
            invalid.pop("timestamp")
        try:
            Reading(**invalid)
        except ValidationError as error:
            print("Validation error:", error)
    coerced = Reading(**{**valid, "temperature_c": "21.5", "humidity": 40})
    print("Coerced:", coerced, type(coerced.temperature_c), type(coerced.humidity))
    try:
        Reading(station_id="X", temperature_c="not a number", humidity=50)
    except ValidationError as error:
        print("Multiple errors:", len(error.errors()))
        for detail in error.errors():
            print(detail["loc"], detail["msg"])
    print("Still valid:", Reading(**valid))
    try:
        Reading(**{**valid, "temperature_c": -50, "humidity": 0})
    except ValidationError as error:
        print("Sensor combination rejected:", error)

if __name__ == "__main__":
    demonstrate()
