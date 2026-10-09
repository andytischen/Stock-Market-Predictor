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

    # market() resolves via the import-time `MARKETS_BY_SYMBOL` table, so the
    # lookup assertion for real registry data stays outside the monkeypatched
    # session-boundary block.
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


def test_calibration_uses_only_preceding_history_for_each_block(monkeypatch):
    probs = pd.Series(np.linspace(0.01, 0.99, 10), index=pd.RangeIndex(10))
    outcomes = pd.Series((np.arange(10) % 2).astype(int), index=probs.index)
    backtest = model.Backtest(probabilities=probs, outcomes=outcomes)

    seen: list[list[int]] = []

    def fake_calibrator(history):
        seen.append(list(history.probabilities.index))
        return lambda values: np.asarray(values, dtype=float)

    monkeypatch.setattr(model, "calibrator", fake_calibrator)

    published = model.calibrated(backtest, min_history=3, step=2)
    assert seen == [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7, 8]]
    assert list(published.probabilities.index) == list(range(3, 10))

    unchanged = model.calibrated(backtest, min_history=500)
    assert unchanged is backtest


def test_calibrator_preserves_order_for_increasing_logits():
    probs = pd.Series([0.01, 0.2, 0.5, 0.8, 0.99], index=pd.RangeIndex(5))
    outcomes = pd.Series([0, 0, 1, 1, 1], index=probs.index)
    backtest = model.Backtest(probabilities=probs, outcomes=outcomes)

    calibrator = model.calibrator(backtest)
    mapped = calibrator(np.array([0.01, 0.5, 0.99]))
    assert mapped[0] < mapped[1] < mapped[2]
    assert mapped[0] < 0.5 < mapped[2]
