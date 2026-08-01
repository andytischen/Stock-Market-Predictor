import pandas as pd
import pytest

from gapmodel import data


@pytest.fixture
def bars() -> pd.DataFrame:
    index = pd.bdate_range("2005-01-03", periods=400)
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0}, index=index)


@pytest.fixture
def counting_download(monkeypatch, bars):
    calls: list[str] = []

    def fake(symbol: str, start: str) -> pd.DataFrame:
        calls.append(start)
        return bars.loc[bars.index >= pd.Timestamp(start)]

    monkeypatch.setattr(data, "_download", fake)
    return calls


def test_warm_cache_is_reused(tmp_path, counting_download):
    first = data.load_symbol("^GSPC", "2005-01-01", tmp_path)
    second = data.load_symbol("^GSPC", "2005-01-01", tmp_path)
    # The first bar is later than the requested start; that must not invalidate.
    assert first.index.min() > pd.Timestamp("2005-01-01")
    assert counting_download == ["2005-01-01"]
    assert second.equals(first)


def test_start_is_honoured_against_a_warm_cache(tmp_path, counting_download):
    data.load_symbol("^GSPC", "2005-01-01", tmp_path)
    trimmed = data.load_symbol("^GSPC", "2006-01-01", tmp_path)
    assert len(counting_download) == 1
    assert trimmed.index.min() >= pd.Timestamp("2006-01-01")


def test_earlier_start_triggers_a_refresh(tmp_path, counting_download):
    data.load_symbol("^GSPC", "2006-01-01", tmp_path)
    data.load_symbol("^GSPC", "2005-01-01", tmp_path)
    assert counting_download == ["2006-01-01", "2005-01-01"]


def test_refresh_forces_a_download(tmp_path, counting_download):
    data.load_symbol("^GSPC", "2005-01-01", tmp_path)
    data.load_symbol("^GSPC", "2005-01-01", tmp_path, refresh=True)
    assert len(counting_download) == 2
