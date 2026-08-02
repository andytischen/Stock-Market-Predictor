"""Social arbitrage: markets where the model probability diverges from what
correlated peers imply.

The routine computes a ``divergence`` for every forecasted market by comparing
the model's own probability to the weighted-average probability of its most
closely correlated historical peers (correlation measured over a recent window
of daily returns).

A large positive divergence means the model rates this market as significantly
more bullish than its historical co-movers would suggest.  A large negative
divergence means the opposite.  These are the ``social arbitrage`` signals:
markets that are priced by the fitted model at odds with the consensus of the
markets they normally move with.

The peer consensus is a simple weighted average:

    p_consensus(i) = Σ_j w_ij * p_j   /   Σ_j w_ij

where w_ij = |ρ_ij|  (the absolute historical return correlation between
market i and peer j), and the sum runs over all peers that (a) have a model
forecast and (b) clear the ``min_corr`` threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import log_return
from .markets import MARKETS, market

# A full trading year of daily bars.
CORRELATION_WINDOW = 250


@dataclass
class ArbSignal:
    """Social arbitrage signal for one market."""

    symbol: str
    name: str
    region: str
    p_model: float
    p_consensus: float
    divergence: float  # p_model − p_consensus; + = model more bullish than peers
    top_peer: str  # name of the most correlated peer
    top_peer_corr: float  # |ρ| with that peer
    top_peer_prob: float  # that peer's model probability


def return_correlations(
    panel: dict[str, pd.DataFrame],
    window: int = CORRELATION_WINDOW,
) -> pd.DataFrame:
    """Pairwise correlation matrix of the most recent ``window`` daily log-returns.

    Only modelled markets that appear in the panel are included.  The matrix is
    indexed and columned by Yahoo symbol.
    """
    series: dict[str, pd.Series] = {}
    for mkt in MARKETS:
        bars = panel.get(mkt.symbol)
        if bars is None or bars["Close"].dropna().empty:
            continue
        series[mkt.symbol] = log_return(bars["Close"].dropna()).tail(window)
    if len(series) < 2:
        raise ValueError("at least two markets with price history are needed")
    return pd.DataFrame(series).dropna(how="all").corr()


def _peer_consensus(
    symbol: str,
    forecasts: dict[str, float],
    correlations: pd.DataFrame,
    min_corr: float,
) -> tuple[float, str, float, float]:
    """Weighted-average probability from correlated peers.

    Returns (consensus, top_peer_symbol, top_peer_abs_corr, top_peer_prob).
    If no qualifying peer exists, each value is NaN / empty string.
    """
    if symbol not in correlations.index:
        return float("nan"), "", float("nan"), float("nan")

    row = correlations.loc[symbol].drop(index=symbol, errors="ignore")
    peers = {
        s: float(row[s])
        for s in row.index
        if s in forecasts and abs(float(row[s])) >= min_corr
    }
    if not peers:
        return float("nan"), "", float("nan"), float("nan")

    weights = {s: abs(rho) for s, rho in peers.items()}
    total = sum(weights.values())
    consensus = sum(weights[s] * forecasts[s] for s in peers) / total
    top = max(peers, key=lambda s: abs(peers[s]))
    return float(consensus), top, float(abs(peers[top])), float(forecasts[top])


def build_social_arb(
    panel: dict[str, pd.DataFrame],
    forecasts: list,  # list[Forecast] — imported lazily to avoid circular import
    window: int = CORRELATION_WINDOW,
    min_corr: float = 0.1,
) -> list[ArbSignal]:
    """Compute the social arbitrage signal for every forecasted market.

    Returns signals sorted by absolute divergence, largest first.
    """
    correlations = return_correlations(panel, window)
    prob_by_symbol = {f.symbol: f.probability_up for f in forecasts}
    signals: list[ArbSignal] = []
    for f in forecasts:
        consensus, top_sym, top_corr, top_prob = _peer_consensus(
            f.symbol, prob_by_symbol, correlations, min_corr
        )
        if np.isnan(consensus):
            continue
        top_name = market(top_sym).name if top_sym else ""
        signals.append(
            ArbSignal(
                symbol=f.symbol,
                name=f.name,
                region=f.region,
                p_model=f.probability_up,
                p_consensus=float(consensus),
                divergence=f.probability_up - float(consensus),
                top_peer=top_name,
                top_peer_corr=float(top_corr),
                top_peer_prob=float(top_prob),
            )
        )
    return sorted(signals, key=lambda s: abs(s.divergence), reverse=True)


def to_frame(signals: list[ArbSignal]) -> pd.DataFrame:
    """Flat DataFrame suitable for printing or CSV export."""
    return pd.DataFrame(
        [
            {
                "market": s.name,
                "symbol": s.symbol,
                "region": s.region,
                "p_model": round(s.p_model, 4),
                "p_consensus": round(s.p_consensus, 4),
                "divergence": round(s.divergence, 4),
                "top_peer": s.top_peer,
                "top_peer_corr": round(s.top_peer_corr, 3),
                "top_peer_prob": round(s.top_peer_prob, 4),
            }
            for s in signals
        ]
    )
