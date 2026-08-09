import pandas as pd
import pytest

from gapmodel.screener import (
    Criteria,
    Reading,
    average_true_range,
    read_metrics,
    render_text,
    to_frame,
)
from gapmodel.screener import screen as run_screen
from gapmodel.universe import read_universe, us_universe

CRITERIA = Criteria(avg_window=5, atr_window=3)


def _bars(
    closes: list[float],
    volumes: list[float] | None = None,
    span: float = 0.01,
) -> pd.DataFrame:
    """Daily bars around ``closes``, each session spanning ``span`` of its close."""
    index = pd.date_range("2025-01-01", periods=len(closes), freq="B")
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": [c * (1 + span / 2) for c in closes],
            "Low": [c * (1 - span / 2) for c in closes],
            "Close": closes,
            "Volume": volumes if volumes is not None else [1e6] * len(closes),
        },
        index=index,
    )
    return frame


def _reading(**overrides) -> Reading:
    fields = {
        "symbol": "AAA",
        "last": 100.0,
        "change": 0.02,
        "volume": 10e6,
        "avg_volume": 6e6,
        "rel_volume": 1.67,
        "atr": 0.03,
        "asof": pd.Timestamp("2026-08-07"),
    }
    return Reading(**{**fields, **overrides})


def test_metrics_read_the_last_session_against_the_ones_before_it():
    closes = [100.0] * 6 + [110.0]
    volumes = [4e6] * 6 + [20e6]
    reading = read_metrics(_bars(closes, volumes), CRITERIA, symbol="AAA")
    assert reading.symbol == "AAA"
    assert reading.last == pytest.approx(110.0)
    assert reading.change == pytest.approx(0.10)
    assert reading.volume == pytest.approx(20e6)
    # Baseline is the five sessions before the one screened, so today's 20m is
    # not in its own average.
    assert reading.avg_volume == pytest.approx(4e6)
    assert reading.rel_volume == pytest.approx(5.0)


def test_relative_volume_excludes_the_screened_session():
    closes = [100.0] * 7
    volumes = [1e6] * 6 + [10e6]
    reading = read_metrics(_bars(closes, volumes), CRITERIA)
    assert reading.rel_volume == pytest.approx(10.0)


def test_atr_counts_gaps_not_only_the_session_range():
    # A flat 1%-range series that gaps 10% on the last session: the gapped range
    # dominates the intraday one.
    closes = [100.0] * 6 + [110.0]
    bars = _bars(closes)
    ranges = average_true_range(bars, window=1)
    assert ranges == pytest.approx(110.0 * 1.005 - 100.0)


def test_atr_is_a_share_of_price():
    closes = [50.0] * 8
    reading = read_metrics(_bars(closes, span=0.04), CRITERIA)
    assert reading.atr == pytest.approx(0.04)


def test_refuses_short_history():
    with pytest.raises(ValueError):
        read_metrics(_bars([100.0] * 4), CRITERIA)


def test_refuses_missing_volume():
    bars = _bars([100.0] * 8).drop(columns=["Volume"])
    with pytest.raises(ValueError):
        read_metrics(bars, CRITERIA)


def test_funnel_narrows_stage_by_stage(monkeypatch):
    #                       price  move   today's vol   30d avg    atr
    universe = {
        "MOVER": ([100.0] * 6 + [102.0], [8e6] * 6 + [20e6], 0.03),  # clears everything
        "PENNY": ([2.0] * 6 + [2.1], [8e6] * 6 + [20e6], 0.03),  # too cheap
        "THIN": ([100.0] * 6 + [102.0], [2e6] * 6 + [6e6], 0.03),  # illiquid
        "QUIET": ([100.0] * 6 + [102.0], [8e6] * 6 + [8e6], 0.03),  # no unusual volume
        "FLAT": ([100.0] * 7, [8e6] * 6 + [20e6], 0.03),  # active but not moving
        "SLEEPY": ([100.0] * 6 + [102.0], [8e6] * 6 + [20e6], 0.005),  # moves too little
    }

    def fake_load(symbol, start, cache_dir, refresh, require=()):
        closes, volumes, span = universe[symbol]
        return _bars(closes, volumes, span)

    monkeypatch.setattr("gapmodel.screener.load_symbol", fake_load)
    result = run_screen(list(universe), criteria=CRITERIA)

    assert [(s.name, s.kept) for s in result.stages] == [
        ("universe", 6),
        ("liquid", 4),
        ("active", 3),
        ("moving", 1),
    ]
    assert [r.symbol for r in result.readings] == ["MOVER"]


def test_dead_tickers_are_skipped_not_fatal(monkeypatch):
    def fake_load(symbol, start, cache_dir, refresh, require=()):
        if symbol == "DEAD":
            raise RuntimeError("no data returned for DEAD")
        return _bars([100.0] * 6 + [102.0], [8e6] * 6 + [20e6], 0.03)

    monkeypatch.setattr("gapmodel.screener.load_symbol", fake_load)
    result = run_screen(["DEAD", "LIVE"], criteria=CRITERIA)
    assert [r.symbol for r in result.readings] == ["LIVE"]


def test_screen_fails_when_nothing_can_be_read(monkeypatch):
    def fake_load(symbol, start, cache_dir, refresh, require=()):
        raise RuntimeError("no data")

    monkeypatch.setattr("gapmodel.screener.load_symbol", fake_load)
    with pytest.raises(RuntimeError):
        run_screen(["DEAD"], criteria=CRITERIA)


def test_survivors_are_sorted_by_relative_volume(monkeypatch):
    volumes = {"BUSY": 30e6, "BUSIER": 60e6, "BUSIEST": 90e6}

    def fake_load(symbol, start, cache_dir, refresh, require=()):
        return _bars([100.0] * 6 + [102.0], [8e6] * 6 + [volumes[symbol]], 0.03)

    monkeypatch.setattr("gapmodel.screener.load_symbol", fake_load)
    result = run_screen(["BUSY", "BUSIEST", "BUSIER"], criteria=CRITERIA)
    assert [r.symbol for r in result.readings] == ["BUSIEST", "BUSIER", "BUSY"]


def test_asof_screens_an_earlier_session(monkeypatch):
    closes = [100.0] * 6 + [102.0] + [90.0]
    volumes = [8e6] * 6 + [20e6] + [20e6]

    def fake_load(symbol, start, cache_dir, refresh, require=()):
        return _bars(closes, volumes, 0.03)

    monkeypatch.setattr("gapmodel.screener.load_symbol", fake_load)
    bars = _bars(closes, volumes, 0.03)
    up_day = bars.index[-2]
    assert [r.symbol for r in run_screen(["AAA"], criteria=CRITERIA, asof=up_day).readings] == [
        "AAA"
    ]
    # The last session is a 12% fall, so the same name fails the movement stage.
    assert run_screen(["AAA"], criteria=CRITERIA).readings == ()


def test_frame_scales_volumes_to_millions_and_moves_to_percent():
    frame = to_frame((_reading(),))
    assert frame.loc[0, "volume_m"] == pytest.approx(10.0)
    assert frame.loc[0, "avg_volume_m"] == pytest.approx(6.0)
    assert frame.loc[0, "change"] == pytest.approx(2.0)
    assert frame.loc[0, "atr_pct"] == pytest.approx(3.0)
    assert frame.loc[0, "asof"] == "2026-08-07"


def test_render_reports_the_funnel_and_says_so_when_empty(monkeypatch):
    def fake_load(symbol, start, cache_dir, refresh, require=()):
        return _bars([100.0] * 7, [8e6] * 7)

    monkeypatch.setattr("gapmodel.screener.load_symbol", fake_load)
    text = render_text(run_screen(["FLAT"], criteria=CRITERIA))
    assert "universe" in text and "moving" in text
    assert "nothing cleared every filter" in text


def test_criteria_reject_unusable_windows():
    with pytest.raises(ValueError):
        Criteria(avg_window=1)
    with pytest.raises(ValueError):
        Criteria(atr_window=1)


def test_universe_is_deduplicated_and_etfs_are_opt_in():
    plain = us_universe()
    assert len(plain) == len(set(plain))
    assert "SPY" not in plain
    assert "SPY" in us_universe(include_etfs=True)


def test_universe_file_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("aapl\n\n# a comment\nmsft  # inline\nAAPL\n", encoding="utf-8")
    assert read_universe(path) == ["AAPL", "MSFT"]


def test_universe_file_must_contain_tickers(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_universe(path)
