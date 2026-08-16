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

The peer consensus is a weighted average of what each peer's forecast implies
for this market:

    p_consensus(i) = sum_j |rho_ij| * p_ij  /  sum_j |rho_ij|

    p_ij = p_j       when rho_ij > 0
    p_ij = 1 - p_j   when rho_ij < 0

The sign matters: a market that historically moves against its peer reads that
peer's bullish call as a bearish one, so the peer's probability is mirrored
before it is averaged.  Only the strength of the relationship, |rho_ij|, sets the
weight, and the sum runs over all peers that (a) have a model forecast and
(b) clear the ``min_corr`` threshold in absolute terms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import log_return
from .markets import MARKETS, market

log = logging.getLogger(__name__)

# A full trading year of daily bars.
CORRELATION_WINDOW = 250

# Shared returns a pair needs before its correlation is trusted.  A market whose
# cached history is short or stale would otherwise earn a large weight from a
# handful of overlapping sessions.
MIN_OVERLAP = 30


@dataclass
class ArbSignal:
    """Social arbitrage signal for one market."""

    symbol: str
    name: str
    region: str
    p_model: float
    p_consensus: float
    # p_model - p_consensus; + = model more bullish than peers. Full precision,
    # used for ranking; to_frame recomputes it from the rounded columns.
    divergence: float
    top_peer: str  # name of the most strongly correlated peer
    top_peer_corr: float  # signed rho with that peer; negative = inverse peer
    top_peer_prob: float  # that peer's own model probability


def return_correlations(
    panel: dict[str, pd.DataFrame],
    window: int = CORRELATION_WINDOW,
) -> pd.DataFrame:
    """Pairwise correlation matrix of the most recent ``window`` daily log-returns.

    Only modelled markets that appear in the panel are included.  The matrix is
    indexed and columned by Yahoo symbol.

    The series are aligned on a shared calendar before the window is taken, so
    ``window`` counts sessions of the combined calendar rather than of whichever
    market happens to have the longest history.  A pair sharing fewer than
    ``MIN_OVERLAP`` returns in that window is left as NaN instead of being
    scored on a handful of observations.
    """
    series: dict[str, pd.Series] = {}
    for mkt in MARKETS:
        bars = panel.get(mkt.symbol)
        if bars is None or bars["Close"].dropna().empty:
            continue
        series[mkt.symbol] = log_return(bars["Close"].dropna()).dropna()
    if len(series) < 2:
        raise ValueError("at least two markets with price history are needed")
    returns = pd.DataFrame(series).sort_index().tail(window)
    return returns.corr(min_periods=min(MIN_OVERLAP, window))


def _peer_consensus(
    symbol: str,
    forecasts: dict[str, float],
    correlations: pd.DataFrame,
    min_corr: float,
) -> tuple[float, str, float, float]:
    """Weighted-average probability implied for ``symbol`` by its correlated peers.

    Returns (consensus, top_peer_symbol, top_peer_signed_corr, top_peer_prob).
    If no qualifying peer exists, each value is NaN / empty string.
    """
    if symbol not in correlations.index:
        return float("nan"), "", float("nan"), float("nan")

    row = correlations.loc[symbol].drop(index=symbol, errors="ignore")
    peers = {
        s: float(row[s]) for s in row.index if s in forecasts and abs(float(row[s])) >= min_corr
    }
    if not peers:
        return float("nan"), "", float("nan"), float("nan")

    # An inversely correlated peer implies the mirror of its own probability.
    implied = {s: forecasts[s] if rho > 0 else 1.0 - forecasts[s] for s, rho in peers.items()}
    weights = {s: abs(rho) for s, rho in peers.items()}
    total = sum(weights.values())
    consensus = sum(weights[s] * implied[s] for s in peers) / total
    top = max(peers, key=lambda s: abs(peers[s]))
    return float(consensus), top, float(peers[top]), float(forecasts[top])


def build_social_arb(
    panel: dict[str, pd.DataFrame],
    forecasts: list,  # list[Forecast] — imported lazily to avoid circular import
    window: int = CORRELATION_WINDOW,
    min_corr: float = 0.1,
) -> list[ArbSignal]:
    """Compute the social arbitrage signal for every forecasted market.

    A market with no peer clearing ``min_corr`` has nothing to diverge from and
    is left out of the result; the names dropped that way are logged so that a
    short — or empty — table can be explained.

    Returns signals sorted by absolute divergence, largest first.
    """
    correlations = return_correlations(panel, window)
    prob_by_symbol = {f.symbol: f.probability_up for f in forecasts}
    signals: list[ArbSignal] = []
    skipped: list[str] = []
    for f in forecasts:
        consensus, top_sym, top_corr, top_prob = _peer_consensus(
            f.symbol, prob_by_symbol, correlations, min_corr
        )
        if np.isnan(consensus):
            skipped.append(f.symbol)
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
    if skipped:
        log.warning(
            "no peer above |rho| %.2f over %d sessions for %s",
            min_corr,
            window,
            ", ".join(sorted(skipped)),
        )
    return sorted(signals, key=lambda s: abs(s.divergence), reverse=True)


def _row(s: ArbSignal) -> dict[str, object]:
    # The divergence is rounded from the two printed probabilities rather than
    # from the full-precision ones, so the column a reader checks by hand is the
    # difference of the two columns either side of it.
    p_model = round(s.p_model, 4)
    p_consensus = round(s.p_consensus, 4)
    return {
        "market": s.name,
        "symbol": s.symbol,
        "region": s.region,
        "p_model": p_model,
        "p_consensus": p_consensus,
        "divergence": round(p_model - p_consensus, 4),
        "top_peer": s.top_peer,
        "top_peer_corr": round(s.top_peer_corr, 3),
        "top_peer_prob": round(s.top_peer_prob, 4),
    }


def to_frame(signals: list[ArbSignal]) -> pd.DataFrame:
    """Flat DataFrame suitable for printing or CSV export."""
    return pd.DataFrame([_row(s) for s in signals])
