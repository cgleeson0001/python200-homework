"""Aggregate validated hourly records into daily summaries."""
from collections import defaultdict
from dataclasses import dataclass
from .records import HourlyReading

@dataclass
class DailySummary:
    """Daily extrema in °C, precipitation in mm, and observation count."""
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the day's temperature spread in °C."""
        return self.temp_max - self.temp_min

class DailyAggregator:
    """Aggregate calendar days having at least min_hours observations."""
    def __init__(self, min_hours: int = 24) -> None:
        """Set the minimum number of observations needed to report a day."""
        if min_hours < 1:
            raise ValueError("min_hours must be positive")
        self.min_hours = min_hours

    def _groups(self, readings: list[HourlyReading]) -> dict[str, list[HourlyReading]]:
        """Group readings by the YYYY-MM-DD portion of timestamp."""
        groups: dict[str, list[HourlyReading]] = defaultdict(list)
        for reading in readings:
            groups[reading.timestamp[:10]].append(reading)
        return dict(groups)

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Return chronologically sorted summaries for sufficiently observed days."""
        result = []
        for date, hours in sorted(self._groups(readings).items()):
            if len(hours) < self.min_hours:
                continue
            temps = [h.temperature_c for h in hours]
            result.append(DailySummary(date, max(temps), min(temps),
                                      sum(h.precipitation_mm for h in hours), len(hours)))
        return result

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """List chronologically the dates skipped for having too few readings."""
        return sorted(date for date, hours in self._groups(readings).items()
                      if len(hours) < self.min_hours)
# --- Intentional Test Failure ---
# I deliberately changed max() to min() in DailyAggregator.summarize().
# The test_extrema test failed because the expected maximum
# temperature was 15, but the function returned 10.
#
# Pytest results: 1 failed, 6 passed in 0.26s.
#
# This demonstrates that the test correctly detects errors
# in the maximum temperature calculation.