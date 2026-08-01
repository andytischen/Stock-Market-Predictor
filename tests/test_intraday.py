import numpy as np
import pandas as pd

from gapmodel.intraday import preopen_features
from gapmodel.markets import market


def hourly_series(start: str = "2024-01-01", hours: int = 24 * 40) -> pd.Series:
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.Series(np.arange(hours, dtype=float) + 1.0, index=index)


def test_features_only_use_bars_completed_before_the_bell():
    close = hourly_series()
    dates = pd.DatetimeIndex(["2024-01-10"])
    frame = preopen_features(market("^GSPC"), dates, {"ES=F": close})

    # The bell is 13:30 UTC; the 12:00 bar closes at 13:00, the 13:00 bar would
    # still be running and must not be used.
    bell_value = close.loc["2024-01-10 12:00":"2024-01-10 12:00"].iloc[0]
    momentum = frame.loc[dates[0], "pre_es_f_momentum"]
    expected_from = close.loc["2024-01-10 09:00":"2024-01-10 09:00"].iloc[0]
    assert momentum == np.log(bell_value / expected_from)


def test_overnight_window_starts_at_the_previous_close():
    close = hourly_series()
    dates = pd.DatetimeIndex(["2024-01-10"])
    frame = preopen_features(market("^GSPC"), dates, {"ES=F": close})

    # Previous cash close is 20:00 UTC the day before; its last complete bar is
    # the one starting at 18:00 (running to 19:00 is complete at 19:00 < 20:00).
    reference = close.loc["2024-01-09 19:00":"2024-01-09 19:00"].iloc[0]
    bell_value = close.loc["2024-01-10 12:00":"2024-01-10 12:00"].iloc[0]
    assert frame.loc[dates[0], "pre_es_f_overnight"] == np.log(bell_value / reference)


def test_dates_before_the_hourly_history_are_nan():
    close = hourly_series(start="2024-01-01")
    frame = preopen_features(market("^GSPC"), pd.DatetimeIndex(["2023-06-01"]), {"ES=F": close})
    assert frame.isna().all().all()
