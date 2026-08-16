import numpy as np
import pandas as pd
import pytest

from gapmodel.social_arb import (
    ArbSignal,
    build_social_arb,
    return_correlations,
    to_frame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYMBOLS = ["^GSPC", "^IXIC", "^N225", "^GDAXI"]


def _bars(n: int = 300, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close),
            "Low": np.minimum(open_, close),
            "Close": close,
        },
        index=dates,
    )


def _independent_panel() -> dict[str, pd.DataFrame]:
    """Each market driven by its own independent noise — correlations near zero."""
    return {sym: _bars(seed=i) for i, sym in enumerate(_SYMBOLS)}


def _correlated_panel(n: int = 300) -> dict[str, pd.DataFrame]:
    """Markets sharing a common factor so pairwise correlations are clearly above 0.1."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    common = rng.normal(0, 0.01, n)  # shared daily return factor
    panel: dict[str, pd.DataFrame] = {}
    for sym in _SYMBOLS:
        idio = rng.normal(0, 0.005, n)
        close = 100 * np.exp(np.cumsum(common + idio))
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        panel[sym] = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(open_, close),
                "Low": np.minimum(open_, close),
                "Close": close,
            },
            index=dates,
        )
    return panel


def _fake_forecast(symbol: str, name: str, prob: float):
    """Minimal stand-in for a Forecast object."""

    class _F:
        pass

    f = _F()
    f.symbol = symbol
    f.name = name
    f.region = "Test"
    f.probability_up = prob
    return f


# ---------------------------------------------------------------------------
# return_correlations
# ---------------------------------------------------------------------------


def test_return_correlations_shape():
    panel = _correlated_panel()
    corr = return_correlations(panel, window=100)
    # Only the four symbols we provided should appear.
    assert set(corr.columns) <= set(panel.keys())
    assert corr.shape[0] == corr.shape[1]


def test_return_correlations_diagonal_is_one():
    panel = _correlated_panel()
    corr = return_correlations(panel, window=100)
    np.testing.assert_allclose(np.diag(corr.values), 1.0)


def test_return_correlations_requires_two_markets():
    panel = {"^GSPC": _bars(seed=0)}
    with pytest.raises(ValueError, match="at least two"):
        return_correlations(panel, window=100)


def test_return_correlations_perfectly_correlated_series():
    dates = pd.bdate_range("2020-01-01", periods=150)
    close = pd.Series(np.cumsum(np.random.default_rng(7).normal(0, 1, 150)) + 100, index=dates)
    bars = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close})
    panel = {"^GSPC": bars, "^IXIC": bars.copy()}
    corr = return_correlations(panel, window=100)
    assert corr.loc["^GSPC", "^IXIC"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_social_arb
# ---------------------------------------------------------------------------


def _make_forecasts(probs: dict[str, float]):
    names = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq Composite",
        "^N225": "Nikkei 225",
        "^GDAXI": "DAX",
    }
    return [_fake_forecast(sym, names[sym], p) for sym, p in probs.items()]


def test_build_social_arb_returns_one_signal_per_market():
    panel = _correlated_panel()
    forecasts = _make_forecasts({"^GSPC": 0.6, "^IXIC": 0.55, "^N225": 0.7, "^GDAXI": 0.5})
    signals = build_social_arb(panel, forecasts, window=100)
    assert len(signals) == len(forecasts)


def test_build_social_arb_sorted_by_abs_divergence():
    panel = _correlated_panel()
    forecasts = _make_forecasts({"^GSPC": 0.6, "^IXIC": 0.55, "^N225": 0.7, "^GDAXI": 0.5})
    signals = build_social_arb(panel, forecasts, window=100)
    divs = [abs(s.divergence) for s in signals]
    assert divs == sorted(divs, reverse=True)


def test_build_social_arb_divergence_is_model_minus_consensus():
    panel = _correlated_panel()
    forecasts = _make_forecasts({"^GSPC": 0.8, "^IXIC": 0.3, "^N225": 0.3, "^GDAXI": 0.3})
    signals = build_social_arb(panel, forecasts, window=100)
    sp500 = next(s for s in signals if s.symbol == "^GSPC")
    assert sp500.divergence == pytest.approx(sp500.p_model - sp500.p_consensus)


def test_build_social_arb_high_model_gives_positive_divergence():
    panel = _correlated_panel()
    # ^GSPC at 0.9 while all peers are at 0.1 — should give a large positive divergence.
    forecasts = _make_forecasts({"^GSPC": 0.9, "^IXIC": 0.1, "^N225": 0.1, "^GDAXI": 0.1})
    signals = build_social_arb(panel, forecasts, window=100)
    sp500 = next(s for s in signals if s.symbol == "^GSPC")
    assert sp500.divergence > 0


def test_build_social_arb_consensus_is_bounded():
    panel = _correlated_panel()
    forecasts = _make_forecasts({"^GSPC": 0.6, "^IXIC": 0.55, "^N225": 0.7, "^GDAXI": 0.5})
    signals = build_social_arb(panel, forecasts, window=100)
    for sig in signals:
        assert 0.0 <= sig.p_consensus <= 1.0


def _inverse_panel(n: int = 300) -> dict[str, pd.DataFrame]:
    """^GSPC and ^IXIC move together; ^N225 and ^GDAXI move against them."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=n)
    common = rng.normal(0, 0.01, n)
    panel: dict[str, pd.DataFrame] = {}
    for sym in _SYMBOLS:
        sign = 1.0 if sym in {"^GSPC", "^IXIC"} else -1.0
        close = 100 * np.exp(np.cumsum(sign * common + rng.normal(0, 0.001, n)))
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        panel[sym] = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(open_, close),
                "Low": np.minimum(open_, close),
                "Close": close,
            },
            index=dates,
        )
    return panel


def test_inverse_peers_are_mirrored_before_averaging():
    panel = _inverse_panel()
    corr = return_correlations(panel, window=200)
    assert corr.loc["^GSPC", "^N225"] < -0.5  # the panel really is inverse

    # Every peer reads bullish. The two that move against ^GSPC therefore imply
    # a bearish ^GSPC, so the consensus must sit below the raw peer average.
    forecasts = _make_forecasts({"^GSPC": 0.5, "^IXIC": 0.9, "^N225": 0.9, "^GDAXI": 0.9})
    signals = build_social_arb(panel, forecasts, window=200, min_corr=0.5)
    sp500 = next(s for s in signals if s.symbol == "^GSPC")
    assert sp500.p_consensus < 0.5
    assert sp500.divergence > 0


def test_top_peer_corr_keeps_the_sign_of_an_inverse_peer():
    panel = _inverse_panel()
    forecasts = _make_forecasts({"^GSPC": 0.5, "^N225": 0.9, "^GDAXI": 0.9})
    signals = build_social_arb(panel, forecasts, window=200, min_corr=0.5)
    sp500 = next(s for s in signals if s.symbol == "^GSPC")
    assert sp500.top_peer_corr < 0


def test_build_social_arb_min_corr_filters_weak_peers():
    panel = _independent_panel()
    forecasts = _make_forecasts({"^GSPC": 0.6, "^IXIC": 0.55, "^N225": 0.7, "^GDAXI": 0.5})
    # Independent random series — correlations near zero, no peer should qualify
    # at min_corr=0.5.
    signals = build_social_arb(panel, forecasts, window=100, min_corr=0.5)
    assert signals == []


# ---------------------------------------------------------------------------
# to_frame
# ---------------------------------------------------------------------------


def test_to_frame_columns():
    sig = ArbSignal(
        symbol="^GSPC",
        name="S&P 500",
        region="Americas",
        p_model=0.7,
        p_consensus=0.5,
        divergence=0.2,
        top_peer="Nasdaq Composite",
        top_peer_corr=0.9,
        top_peer_prob=0.5,
    )
    frame = to_frame([sig])
    expected = {
        "market",
        "symbol",
        "region",
        "p_model",
        "p_consensus",
        "divergence",
        "top_peer",
        "top_peer_corr",
        "top_peer_prob",
    }
    assert expected <= set(frame.columns)
    assert len(frame) == 1


def test_to_frame_divergence_is_the_difference_of_the_printed_columns():
    # Rounding the full-precision difference gives 0.2749; the printed columns,
    # 0.8059 and 0.5309, differ by 0.2750. The table has to say the latter.
    sig = ArbSignal(
        symbol="^GSPC",
        name="S&P 500",
        region="Americas",
        p_model=0.805869,
        p_consensus=0.530924,
        divergence=0.805869 - 0.530924,
        top_peer="Nasdaq Composite",
        top_peer_corr=0.9,
        top_peer_prob=0.5,
    )
    row = to_frame([sig]).iloc[0]
    assert (row["p_model"], row["p_consensus"]) == (0.8059, 0.5309)
    assert row["divergence"] == 0.275
    assert round(sig.divergence, 4) == 0.2749  # what the column used to print


def test_to_frame_empty():
    assert to_frame([]).empty


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_social_arb_command_in_parser():
    from gapmodel.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["social-arb"])
    assert hasattr(args, "func")
    assert args.window == 250  # default CORRELATION_WINDOW


def test_social_arb_command_window_flag():
    from gapmodel.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["social-arb", "--window", "100"])
    assert args.window == 100


def test_social_arb_command_csv_flag():
    from gapmodel.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["social-arb", "--csv", "out.csv"])
    assert args.csv == "out.csv"


@pytest.mark.parametrize("value", ["0", "-5"])
def test_social_arb_command_rejects_non_positive_window(value):
    from gapmodel.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["social-arb", "--window", value])
