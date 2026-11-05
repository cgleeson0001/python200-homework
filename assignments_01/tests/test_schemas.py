import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from weatherkit import HourlyBlock, WeatherResponse

@pytest.fixture
def sample():
    return {"latitude": 35.2, "longitude": -80.8, "timezone": "GMT", "elevation": 250.0,
            "hourly": {"time": ["t1", "t2", "t3"], "temperature_2m": [1, 2, 3], "precipitation": [0, 0.1, 0]}}

def test_real_file():
    # __file__ anchors the path to this test, not the current working directory.
    path = Path(__file__).parent.parent / "weather_raw.json"
    response = WeatherResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert len(response.hourly.time) == 168

def test_invalid_latitude(sample):
    sample["latitude"] = 200.0
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(sample)

def test_unequal_arrays(sample):
    sample["hourly"]["temperature_2m"].pop()
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(sample)

def test_null_temperature(sample):
    sample["hourly"]["temperature_2m"][0] = None
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(sample)
