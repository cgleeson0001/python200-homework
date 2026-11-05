"""Validate untrusted weather API payloads at the package boundary."""
from pydantic import BaseModel, Field, model_validator
from typing import Self

class HourlyBlock(BaseModel):
    """Parallel hourly timestamp, Celsius temperature, and mm precipitation arrays."""
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def equal_lengths(self) -> Self:
        """Reject inconsistent column lengths before records are assembled."""
        if len(self.time) != len(self.temperature_2m) or len(self.time) != len(self.precipitation):
            raise ValueError("hourly arrays must have equal lengths")
        return self

class WeatherResponse(BaseModel):
    """Validated top-level response from the Open-Meteo hourly API."""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
