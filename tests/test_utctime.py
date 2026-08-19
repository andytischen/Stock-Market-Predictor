import pytest

from gapmodel.utctime import format_utc_time, parse_utc_time


@pytest.mark.parametrize(
    ("value", "hours"),
    [
        ("05:00", 5.0),
        ("5:00", 5.0),
        ("23:30", 23.5),
        (" 09:15 ", 9.25),
        ("05:00:00", 5.0),
        ("05:00:36", 5.01),
    ],
)
def test_a_time_of_day_is_hours_from_midnight(value, hours):
    assert parse_utc_time(value) == pytest.approx(hours)


@pytest.mark.parametrize("value", ["2024-01-01", "24:00", "99:00", "5", "05:60", "05:00:60", ""])
def test_anything_that_is_not_a_time_of_day_is_refused(value):
    with pytest.raises(ValueError, match="not a time of day"):
        parse_utc_time(value)


@pytest.mark.parametrize(
    ("hours", "formatted"),
    [
        (5.0, "05:00"),
        (23.5, "23:30"),
        (0.0, "00:00"),
        # A minute short of the hour must not round to a sixtieth minute.
        (13.994, "14:00"),
        (23.999, "23:59"),
    ],
)
def test_hours_format_back_to_a_time_a_browser_accepts(hours, formatted):
    assert format_utc_time(hours) == formatted
