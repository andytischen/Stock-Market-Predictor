"""Probability model for the next opening auction of major stock indices."""

from .markets import INDICATORS, MARKETS
from .predict import Forecast, forecast_all, forecast_market

__all__ = ["INDICATORS", "MARKETS", "Forecast", "forecast_all", "forecast_market"]
__version__ = "0.1.0"
