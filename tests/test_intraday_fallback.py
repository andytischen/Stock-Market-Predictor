import numpy as np
import pandas as pd
import pytest

from gapmodel import intraday
from gapmodel.intraday import load_hourly


def bars(start: pd.Timestamp, periods: int, freq: str, first: float = 1.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"Close": np.arange(periods, dtype=float) + first}, index=index)


@pytest.fixture
def stale_hourly() -> pd.Timestamp:
    """The hour of the last hourly bar, well beyond the staleness tolerance."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    return now - 6 * intraday.BAR_DURATION


def feeds(stale_hourly: pd.Timestamp, fine_interval: str = "30m"):
    """A stale hourly endpoint alongside a finer one that is up to date."""
    hourly = bars(stale_hourly - 47 * intraday.BAR_DURATION, 48, "h")
    fine_start = stale_hourly - intraday.BAR_DURATION
    fine = bars(fine_start, 12, "30min", first=100.0)
    calls: list[str] = []

    def download(symbol, **kwargs):
        interval = kwargs["interval"]
        calls.append(interval)
        if interval == "1h":
            return hourly
        if interval == fine_interval:
            return fine
        return pd.DataFrame()

    return download, calls, hourly, fine


def test_stale_hourly_feed_is_extended_from_finer_bars(tmp_path, monkeypatch, stale_hourly):
    download, calls, hourly, fine = feeds(stale_hourly)
    monkeypatch.setattr(intraday.yf, "download", download)

    close = load_hourly("ES=F", cache_dir=tmp_path)

    assert "30m" in calls
    assert close.index[-1] > hourly.index[-1]
    # The extension is on the hourly grid and carries the finer feed's prices.
    assert (close.index[1:] - close.index[:-1]).max() == intraday.BAR_DURATION
    assert close.iloc[-1] in set(fine["Close"])


def test_a_fresh_hourly_feed_is_left_alone(tmp_path, monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    hourly = bars(now - 47 * intraday.BAR_DURATION, 48, "h")
    calls: list[str] = []

    def download(symbol, **kwargs):
        calls.append(kwargs["interval"])
        return hourly

    monkeypatch.setattr(intraday.yf, "download", download)
    close = load_hourly("ES=F", cache_dir=tmp_path)

    assert calls == ["1h"]
    assert close.index[-1] == hourly.index[-1]


def test_finer_feeds_no_fresher_leave_the_series_unchanged(tmp_path, monkeypatch, stale_hourly):
    hourly = bars(stale_hourly - 47 * intraday.BAR_DURATION, 48, "h")

    def download(symbol, **kwargs):
        if kwargs["interval"] == "1h":
            return hourly
        # Every finer feed is stuck at the same moment as the hourly one.
        return bars(stale_hourly - intraday.BAR_DURATION, 2, "30min")

    monkeypatch.setattr(intraday.yf, "download", download)
    close = load_hourly("ES=F", cache_dir=tmp_path)

    assert close.index[-1] == hourly.index[-1]


def test_a_gap_too_wide_for_the_finer_feeds_is_not_bridged(tmp_path, monkeypatch):
    """Splicing across a hole the finer history cannot cover would fabricate one."""
    end = pd.Timestamp.now(tz="UTC").floor("h") - intraday.MAX_FALLBACK_GAP - pd.Timedelta(days=1)
    hourly = bars(end - 47 * intraday.BAR_DURATION, 48, "h")
    calls: list[str] = []

    def download(symbol, **kwargs):
        calls.append(kwargs["interval"])
        return hourly

    monkeypatch.setattr(intraday.yf, "download", download)
    load_hourly("ES=F", cache_dir=tmp_path)

    assert calls == ["1h"]


def test_the_next_interval_is_tried_when_one_fails(tmp_path, monkeypatch, stale_hourly):
    hourly = bars(stale_hourly - 47 * intraday.BAR_DURATION, 48, "h")
    fine = bars(stale_hourly - intraday.BAR_DURATION, 12, "15min", first=100.0)

    def download(symbol, **kwargs):
        interval = kwargs["interval"]
        if interval == "1h":
            return hourly
        if interval == "30m":
            raise RuntimeError("no 30m data")
        return fine

    monkeypatch.setattr(intraday.yf, "download", download)
    close = load_hourly("ES=F", cache_dir=tmp_path)

    assert close.index[-1] > hourly.index[-1]
