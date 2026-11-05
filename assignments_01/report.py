"""Command-line report for the 7-day Open-Meteo fixture."""
import json
from pathlib import Path
from weatherkit import DailyAggregator, WeatherResponse, to_readings

def main() -> None:
    """Validate source data, aggregate records, and print a daily report."""
    source = Path(__file__).parent / "weather_raw.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    response = WeatherResponse.model_validate(payload)
    readings = to_readings(response)
    aggregator = DailyAggregator()
    print(f"Latitude: {response.latitude}; timezone: {response.timezone}; hours: {len(readings)}")
    print(f"{'Date':<12}{'High °C':>10}{'Low °C':>10}{'Rain mm':>12}{'Range °C':>12}")
    for s in aggregator.summarize(readings):
        print(f"{s.date:<12}{s.temp_max:>10.1f}{s.temp_min:>10.1f}"
              f"{s.precipitation_sum:>12.1f}{s.temp_range():>12.1f}")
    incomplete = aggregator.incomplete_days(readings)
    if incomplete:
        print("WARNING: Incomplete days dropped: " + ", ".join(incomplete))

if __name__ == "__main__":
    main()
# Without this guard, importing report.py would automatically run main()
# and unexpectedly read a file and print output for the importing program.

# --- Task 7: Reflection (review and personalize before submission) ---
# 1. Strict validation is useful when missing temperatures would make maximum
#    or minimum calculations misleading. For exploratory reports, a gap may be
#    acceptable: change temperature_2m to list[float | None], then explicitly
#    skip or account for missing measurements in downstream transformations.
# 2. At noon, the current day has fewer than 24 readings, so it is omitted.
#    incomplete_days() identifies the missing date for monitoring and warnings.
# 3. In Week 10, another pipeline can import WeatherResponse, to_readings,
#    and DailyAggregator independently without running the CLI report.
