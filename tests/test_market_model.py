from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gapmodel import markets, model, universe


def test_market_lookup_and_session_boundary_are_safe(monkeypatch):
    monkeypatch.setattr(
        markets,
        "MARKETS",
        (
            SimpleNamespace(close_utc=0.5),
            SimpleNamespace(close_utc=8.0),
        ),
    )
    monkeypatch.setattr(markets, "INDICATORS", (SimpleNamespace(close_utc=12.0),))
    monkeypatch.setattr(markets, "CURVE_CLOSE_UTC", 18.0)
    monkeypatch.setattr(markets, "FUNDS_CLOSE_UTC", 21.0)
    monkeypatch.setattr(markets, "BILL_CLOSE_UTC", 6.0)

    assert markets.lag_days(8.0, 9.0) == 0
    assert markets.lag_days(12.0, 9.0) == 1
    assert markets.last_observed_utc(9.0) == pytest.approx(8.0)
    assert markets.last_observed_utc(18.0) == pytest.approx(12.0)

    assert markets.market("^FTSE").open_source == "ISF.L"
    with pytest.raises(KeyError, match="unknown market"):
        markets.market("NOT_A_MARKET")


def test_universe_files_strip_comments_and_deduplicate(tmp_path):
    path = tmp_path / "symbols.txt"
    path.write_text(
        "  aapl  # keeps the first name\nMSFT\nAAPL\n# comment\n\nmsft\n",
        encoding="utf-8",
    )

    assert universe.read_universe(path) == ["AAPL", "MSFT"]
    assert universe.us_universe(include_etfs=True)[:3] == ["AAPL", "ABBV", "ABT"]
    assert "QQQ" in universe.us_universe(include_etfs=True)
    assert "QQQ" not in universe.us_universe()
    assert universe.modelled_universe()[:5] == ["AAPL", "ADBE", "AMD", "AMGN", "AMZN"]
    assert len(universe.modelled_universe()) == len(set(universe.modelled_universe()))


def test_walk_forward_and_window_metrics_use_only_oos_rows():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    signal = np.arange(len(dates), dtype=float)
    features = pd.DataFrame({"signal": signal}, index=dates)
    labels = pd.Series((np.arange(len(dates)) % 2).astype(int), index=dates)

    backtest = model.walk_forward(features, labels, min_train=10, refit_every=10)

    assert len(backtest.probabilities) == 70
    assert backtest.probabilities.index[0] == dates[10]
    assert backtest.probabilities.between(0.0, 1.0).all()
    assert backtest.outcomes.equals(labels.iloc[10:80])

    metrics = backtest.window_metrics(since="2024-01-15", until="2024-01-25")
    assert metrics["n"] == 11
    with pytest.raises(ValueError, match="no out-of-sample predictions"):
        backtest.window_metrics(since="2100-01-01")


def test_calibration_uses_the_walk_forward_record_without_leaking_lookahead(monkeypatch):
    probs = pd.Series(np.linspace(0.01, 0.99, 10), index=pd.RangeIndex(10))
    outcomes = pd.Series((np.arange(10) % 2).astype(int), index=probs.index)
    backtest = model.Backtest(probabilities=probs, outcomes=outcomes)

    actual_calibrator = model.calibrator
    calibrator = actual_calibrator(backtest)
    mapped = calibrator(np.array([0.01, 0.5, 0.99]))
    assert np.all(np.isfinite(mapped))
    assert mapped.min() >= 0.0
    assert mapped.max() <= 1.0
    assert mapped[1] == pytest.approx(0.5, abs=0.5)

    seen: list[int] = []

    def fake_calibrator(history):
        seen.append(history.probabilities.index[-1])
        return lambda values: np.asarray(values, dtype=float)

    monkeypatch.setattr(model, "calibrator", fake_calibrator)

    published = model.calibrated(backtest, min_history=3, step=2)
    assert seen == [2, 4, 6, 8]
    assert list(published.probabilities.index) == [3, 4, 5, 6, 7, 8, 9]

    unchanged = model.calibrated(backtest, min_history=500)
    assert unchanged is backtest
