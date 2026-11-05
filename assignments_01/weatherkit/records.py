"""Internal typed records created after API boundary validation."""
from dataclasses import dataclass
from .schemas import WeatherResponse

@dataclass
class HourlyReading:
    """One hourly observation, temperature in °C and precipitation in mm."""
    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert validated parallel arrays to time-ordered records.

    Args:
        response: A previously validated hourly weather response.

    Returns:
        One HourlyReading per hourly timestamp, in original order.
    """
    hourly = response.hourly
    # Pydantic validated the untrusted input at the boundary. These internal
    # records are dataclasses because repeating validation is unnecessary.
    return [HourlyReading(t, temp, precip) for t, temp, precip in zip(
        hourly.time, hourly.temperature_2m, hourly.precipitation
    )]
