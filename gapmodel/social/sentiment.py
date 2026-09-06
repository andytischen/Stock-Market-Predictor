"""Thin wrapper around vaderSentiment for single-text scoring.

If ``vaderSentiment`` is not installed every text scores 0.0 so the rest of
the social pipeline degrades gracefully rather than crashing.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _Analyzer

    _analyzer: _Analyzer | None = _Analyzer()
except Exception:  # pragma: no cover - only hit when package is absent
    _analyzer = None
    log.warning(
        "vaderSentiment is not installed; all social sentiment scores will be 0.0. "
        "Install it with: pip install vaderSentiment"
    )


def score_text(text: str) -> float:
    """Return the VADER compound sentiment score for *text* in [-1, 1].

    Returns 0.0 when vaderSentiment is unavailable or *text* is empty.
    """
    if not text or _analyzer is None:
        return 0.0
    return float(_analyzer.polarity_scores(text)["compound"])
