import pytest
from weatherkit import HourlyReading, WeatherResponse, to_readings

@pytest.fixture
def response():
    return WeatherResponse.model_validate({
        "latitude": 35, "longitude": -80, "timezone": "GMT", "elevation": 250,
        "hourly": {"time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
                   "temperature_2m": [11, 12, 13], "precipitation": [0, 0.2, 0.3]}})

def test_order_and_count(response):
    readings = to_readings(response)
    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"

@pytest.mark.parametrize("i", [0, 1, 2])
def test_column_alignment(response, i):
    record = to_readings(response)[i]
    assert record.timestamp == response.hourly.time[i]
    assert record.temperature_c == response.hourly.temperature_2m[i]
    assert record.precipitation_mm == response.hourly.precipitation[i]

def test_dataclass_equality():
    assert HourlyReading("t", 1.0, 0.0) == HourlyReading("t", 1.0, 0.0)
