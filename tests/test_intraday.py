import os
import time

import numpy as np
import pandas as pd

from gapmodel import intraday
from gapmodel.intraday import load_hourly, preopen_features
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


def test_dates_beyond_the_hourly_history_are_nan_not_zero():
    close = hourly_series(start="2024-01-01", hours=24 * 5)
    late = pd.DatetimeIndex(["2024-01-20"])  # well past the last bar
    frame = preopen_features(market("^GSPC"), late, {"ES=F": close})
    assert frame.isna().all().all()


def test_recent_but_incomplete_history_still_measures_a_move():
    # Ends at 20:00 the evening before: short of the bell but past the previous
    # close, so the overnight window is real, if truncated.
    close = hourly_series(start="2024-01-01", hours=24 * 8 + 21)
    frame = preopen_features(market("^GSPC"), pd.DatetimeIndex(["2024-01-10"]), {"ES=F": close})
    assert (frame["pre_es_f_overnight"] != 0).all()


def test_a_window_collapsing_onto_one_bar_is_nan():
    # Ends at the previous close itself: both ends of the overnight window
    # resolve to the same bar, which must read as unknown, not as "no move".
    close = hourly_series(start="2024-01-01", hours=24 * 8 + 20)
    frame = preopen_features(market("^GSPC"), pd.DatetimeIndex(["2024-01-10"]), {"ES=F": close})
    assert frame["pre_es_f_overnight"].isna().all()


def test_dates_before_the_hourly_history_are_nan():
    close = hourly_series(start="2024-01-01")
    frame = preopen_features(market("^GSPC"), pd.DatetimeIndex(["2023-06-01"]), {"ES=F": close})
    assert frame.isna().all().all()


def test_hourly_cache_expires(tmp_path, monkeypatch):
    calls = []

    def fake_download(symbol, **kwargs):
        calls.append(symbol)
        index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        return pd.DataFrame({"Close": np.arange(48, dtype=float) + 1.0}, index=index)

    monkeypatch.setattr(intraday.yf, "download", fake_download)

    load_hourly("ES=F", cache_dir=tmp_path)
    load_hourly("ES=F", cache_dir=tmp_path)
    assert len(calls) == 1  # still fresh

    stale = time.time() - intraday.CACHE_TTL.total_seconds() - 60
    os.utime(intraday._cache_path(tmp_path, "ES=F"), (stale, stale))
    load_hourly("ES=F", cache_dir=tmp_path)
    assert len(calls) == 2


def test_bars_without_a_price_are_dropped(tmp_path, monkeypatch):
    index = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame({"Close": [1.0, np.nan, 3.0, 4.0]}, index=index)
    monkeypatch.setattr(intraday.yf, "download", lambda symbol, **kwargs: frame)
    assert len(load_hourly("ES=F", cache_dir=tmp_path)) == 3
