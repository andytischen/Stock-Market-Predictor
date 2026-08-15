import numpy as np
import pandas as pd
import pytest

from gapmodel.model import Backtest
from gapmodel.scorecard import (
    DRIFT_MIN_SESSIONS,
    Call,
    _record,
    append_log,
    build_scorecard,
    calls_frame,
    drifting,
    realised_gaps,
    render_text,
    to_frame,
)


def _sessions(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-05", periods=n)


def _backtest(probabilities: list[float], outcomes: list[int]) -> Backtest:
    index = _sessions(len(probabilities))
    prob = pd.Series(probabilities, index=index)
    out = pd.Series(outcomes, index=index)
    return Backtest(probabilities=prob, outcomes=out, metrics={"brier_skill": 0.30})


def _gaps(index: pd.DatetimeIndex, outcomes: list[int]) -> pd.Series:
    # A one-percent move with the sign of the realised outcome.
    return pd.Series([np.log(1.01) if y else np.log(0.99) for y in outcomes], index=index)


def test_the_window_keeps_only_the_last_sessions():
    probabilities = [0.6] * 30
    outcomes = [1] * 30
    result = _backtest(probabilities, outcomes)
    record = _record("^GSPC", result, _gaps(result.probabilities.index, outcomes), window=5)

    assert len(record.calls) == 5
    assert record.latest.session == result.probabilities.index[-1]
    assert record.window["n"] == 5
    # The full-sample metrics are carried unchanged beside the window's.
    assert record.full["brier_skill"] == pytest.approx(0.30)


def test_a_call_is_scored_against_the_realised_direction():
    index = _sessions(2)
    up = Call("^GSPC", index[0], probability=0.7, gap=np.log(1.01), outcome=1)
    down = Call("^GSPC", index[1], probability=0.7, gap=np.log(0.99), outcome=0)

    assert up.direction == "up" and up.realised == "up" and up.hit
    assert down.direction == "up" and down.realised == "down" and not down.hit
    assert up.gap_pct == pytest.approx(1.0, abs=1e-9)
    assert down.gap_pct == pytest.approx(-1.0, abs=1e-9)


def test_a_window_worse_than_its_base_rate_is_flagged_as_drifting():
    # Confidently wrong on every session: the base rate would have scored better.
    n = DRIFT_MIN_SESSIONS + 5
    outcomes = [1, 0] * n
    probabilities = [0.05 if y else 0.95 for y in outcomes]
    result = _backtest(probabilities, outcomes)
    record = _record("^GSPC", result, _gaps(result.probabilities.index, outcomes), window=n)

    assert record.window["brier_skill"] < 0
    assert record.drifting
    assert record.skill_change < 0
    assert drifting([record]) == [record]
    assert "worse calibrated than the base rate" in render_text([record], window=n)


def test_a_short_window_cannot_raise_the_flag_on_its_own():
    """Brier skill over a handful of sessions swings on single outcomes."""
    n = DRIFT_MIN_SESSIONS - 1
    outcomes = ([1, 0] * n)[:n]
    probabilities = [0.05 if y else 0.95 for y in outcomes]
    result = _backtest(probabilities, outcomes)
    record = _record("^GSPC", result, _gaps(result.probabilities.index, outcomes), window=n)

    assert record.window["brier_skill"] < 0
    assert not record.drifting
    assert drifting([record]) == []


def test_a_good_window_is_not_flagged():
    outcomes = [1, 0] * DRIFT_MIN_SESSIONS
    probabilities = [0.9 if y else 0.1 for y in outcomes]
    result = _backtest(probabilities, outcomes)
    gaps = _gaps(result.probabilities.index, outcomes)
    record = _record("^GSPC", result, gaps, window=len(outcomes))

    assert record.window["brier_skill"] > 0
    assert not record.drifting
    assert record.hits == len(outcomes)


def test_a_non_positive_window_is_refused():
    result = _backtest([0.6] * 10, [1] * 10)
    with pytest.raises(ValueError, match="at least one session"):
        _record("^GSPC", result, _gaps(result.probabilities.index, [1] * 10), window=0)


def test_the_tables_carry_one_row_per_market_and_one_per_session():
    outcomes = [1, 0, 1, 0, 1]
    result = _backtest([0.6, 0.4, 0.55, 0.45, 0.7], outcomes)
    record = _record("^GSPC", result, _gaps(result.probabilities.index, outcomes), window=5)

    summary = to_frame([record])
    assert len(summary) == 1
    assert summary.loc[0, "symbol"] == "^GSPC"
    assert summary.loc[0, "n"] == 5

    calls = calls_frame([record])
    assert len(calls) == 5
    assert list(calls["session"]) == sorted(calls["session"])
    assert set(calls.columns) == {
        "session",
        "symbol",
        "p_open_up",
        "called",
        "realised",
        "gap_pct",
        "hit",
    }


def test_the_log_keeps_one_row_per_session_when_runs_overlap(tmp_path):
    outcomes = [1, 0, 1, 0, 1, 1]
    result = _backtest([0.6, 0.4, 0.55, 0.45, 0.7, 0.65], outcomes)
    gaps = _gaps(result.probabilities.index, outcomes)
    path = tmp_path / "log" / "scorecard.csv"

    first = _record("^GSPC", result, gaps, window=4)
    append_log([first], path)
    # A later run whose window overlaps everything already logged.
    second = _record("^GSPC", result, gaps, window=6)
    merged = append_log([second], path)

    assert len(merged) == 6
    assert merged["session"].is_unique
    assert len(pd.read_csv(path)) == 6


def test_a_rescored_session_replaces_its_old_row(tmp_path):
    outcomes = [1, 1]
    result = _backtest([0.6, 0.6], outcomes)
    gaps = _gaps(result.probabilities.index, outcomes)
    path = tmp_path / "scorecard.csv"
    append_log([_record("^GSPC", result, gaps, window=2)], path)

    corrected = _backtest([0.6, 0.6], [1, 0])
    merged = append_log(
        [_record("^GSPC", corrected, _gaps(corrected.probabilities.index, [1, 0]), window=2)],
        path,
    )

    last = merged[merged["session"] == merged["session"].max()].iloc[0]
    assert last["realised"] == "down"
    assert len(merged) == 2


def test_realised_gaps_correct_a_single_name_for_dividends():
    index = _sessions(3)
    bars = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "Close": [100.0, 100.0, 100.0],
            # A dividend paid on the last session: Adj Close lags the print.
            "Adj Close": [99.0, 99.0, 100.0],
        },
        index=index,
    )
    panel = {"MU": bars}
    gaps = realised_gaps("MU", panel)

    # Without the correction the ex-dividend morning reads as a flat open; with
    # it the previous close is scaled down and the gap is the dividend back.
    assert gaps.iloc[-1] > 0
    assert gaps.iloc[-1] == pytest.approx(np.log(100.0 / 99.0))


def test_an_unmodellable_market_is_skipped_rather_than_fatal(monkeypatch, caplog):
    from gapmodel import scorecard as scorecard_mod

    def only_the_second(symbol, panel, **kwargs):
        if symbol == "^GSPC":
            raise ValueError("only 12 usable rows")
        outcomes = [1, 0, 1]
        result = _backtest([0.6, 0.4, 0.6], outcomes)
        return _record(symbol, result, _gaps(result.probabilities.index, outcomes), window=3)

    monkeypatch.setattr(scorecard_mod, "score", only_the_second)
    with caplog.at_level("WARNING"):
        records = build_scorecard({}, symbols=["^GSPC", "^N225"])

    assert [r.symbol for r in records] == ["^N225"]
    assert "no scorecard for ^GSPC" in caplog.text


def test_every_market_failing_is_an_error(monkeypatch):
    from gapmodel import scorecard as scorecard_mod

    def fail(symbol, panel, **kwargs):
        raise ValueError("nope")

    monkeypatch.setattr(scorecard_mod, "score", fail)
    with pytest.raises(RuntimeError, match="no market could be scored"):
        build_scorecard({}, symbols=["^GSPC"])
