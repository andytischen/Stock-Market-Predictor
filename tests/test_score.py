import numpy as np
import pandas as pd
import pytest

from gapmodel.score import DEFAULT_WINDOW, TrendScore, to_frame, trend_score


def _frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=index)


def test_trend_score_is_zscore_of_last_close():
    closes = [10.0, 12.0, 14.0, 16.0, 18.0]
    tail = np.array(closes)
    expected = (closes[-1] - tail.mean()) / tail.std(ddof=1)
    assert trend_score(_frame(closes), window=len(closes)) == pytest.approx(expected)


def test_positive_when_last_above_average_negative_when_below():
    up = _frame([1, 2, 3, 4, 5, 6, 7, 8, 9, 20.0])
    down = _frame([20, 9, 8, 7, 6, 5, 4, 3, 2, 1.0])
    assert trend_score(up, window=10) > 0
    assert trend_score(down, window=10) < 0


def test_only_the_trailing_window_matters():
    early = [1.0] * 300
    recent = [1.0, 1.1, 1.2, 1.3, 1.4]
    frame = _frame(early + recent)
    windowed = trend_score(frame, window=len(recent))
    assert windowed == pytest.approx(trend_score(_frame(recent), window=len(recent)))


def test_refuses_short_history():
    with pytest.raises(ValueError):
        trend_score(_frame([1.0, 2.0, 3.0]), window=DEFAULT_WINDOW)


def test_refuses_flat_series():
    with pytest.raises(ValueError):
        trend_score(_frame([5.0] * 10), window=10)


def test_to_frame_sorts_and_rounds():
    scores = [
        TrendScore("AAA", 1.234, 10.0, pd.Timestamp("2026-08-04"), 200),
        TrendScore("BBB", -2.0, 5.561, pd.Timestamp("2026-08-04"), 200),
    ]
    frame = to_frame(scores)
    assert list(frame["symbol"]) == ["AAA", "BBB"]
    assert frame.loc[0, "score"] == pytest.approx(1.23)
    assert frame.loc[1, "last"] == pytest.approx(5.56)
    assert frame.loc[1, "score"] == pytest.approx(-2.0)
    assert list(frame["asof"]) == ["2026-08-04", "2026-08-04"]
