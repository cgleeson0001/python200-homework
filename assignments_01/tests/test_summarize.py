import pytest
from weatherkit import HourlyReading, DailyAggregator, DailySummary

@pytest.fixture
def readings():
    return [HourlyReading("2026-04-08T00:00", 10, 0.1),
            HourlyReading("2026-04-08T01:00", 15, 0.2),
            HourlyReading("2026-04-09T00:00", 3, 0.3)]

def test_groups_into_two_dates(readings):
    assert [s.date for s in DailyAggregator(min_hours=1).summarize(readings)] == ["2026-04-08", "2026-04-09"]

def test_extrema(readings):
    summary = DailyAggregator(min_hours=1).summarize(readings)[0]
    assert summary.temp_max == 15 and summary.temp_min == 10
    assert summary.temp_range() == 5

def test_precipitation(readings):
    assert DailyAggregator(min_hours=1).summarize(readings)[0].precipitation_sum == pytest.approx(0.3)

def test_incomplete_day_dropped(readings):
    agg = DailyAggregator(min_hours=2)
    assert [s.date for s in agg.summarize(readings)] == ["2026-04-08"]
    assert agg.incomplete_days(readings) == ["2026-04-09"]

@pytest.mark.parametrize("minimum,expected_count", [(1, 2), (2, 1), (3, 0)])
def test_threshold_parameter(readings, minimum, expected_count):
    assert len(DailyAggregator(min_hours=minimum).summarize(readings)) == expected_count

# Mutation check to perform locally: temporarily change the temperature max()
# in summarize.py to min(). The test_extrema test should fail. Restore max().
