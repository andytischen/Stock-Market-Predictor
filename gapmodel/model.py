"""Probability model: regularised logistic regression with a walk-forward backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_TRAIN = 500
REFIT_EVERY = 21


def make_pipeline(c: float = 0.1) -> Pipeline:
    """Standardised L2 logistic regression; the small C keeps weights honest."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c, penalty="l2", solver="lbfgs", max_iter=2000, class_weight=None
                ),
            ),
        ]
    )


@dataclass
class Backtest:
    """Out-of-sample results of the walk-forward evaluation."""

    probabilities: pd.Series
    outcomes: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)

    def reliability(self, bins: int = 5) -> pd.DataFrame:
        """Predicted vs realised frequency, the practical calibration check."""
        edges = np.linspace(0.0, 1.0, bins + 1)
        bucket = pd.cut(self.probabilities, edges, include_lowest=True)
        grouped = pd.DataFrame(
            {"p": self.probabilities, "y": self.outcomes, "bucket": bucket}
        ).groupby("bucket", observed=True)
        return grouped.agg(predicted=("p", "mean"), realised=("y", "mean"), count=("y", "size"))

    def window_metrics(
        self,
        since: pd.Timestamp | None = None,
        until: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """Metrics computed over a date sub-window of the OOS predictions.

        Dates are compared against the session index of the backtest
        probabilities, so ``since`` and ``until`` should be plain dates or
        timestamps without timezone info (matching how the panel is indexed).
        """
        mask = pd.Series(True, index=self.probabilities.index)
        if since is not None:
            mask &= self.probabilities.index >= since
        if until is not None:
            mask &= self.probabilities.index <= until
        prob = self.probabilities.loc[mask]
        out = self.outcomes.loc[mask]
        if len(prob) == 0:
            raise ValueError("no out-of-sample predictions in the requested window")
        return _metrics(prob.to_numpy(), out.to_numpy())


def _metrics(prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    base_rate = float(y.mean())
    baseline_brier = float(np.mean((base_rate - y) ** 2))
    brier = float(brier_score_loss(y, prob))
    return {
        "n": float(len(y)),
        "base_rate": base_rate,
        "accuracy": float(np.mean((prob > 0.5) == (y == 1))),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": brier,
        # Positive skill means the model beats always predicting the base rate.
        "brier_skill": float(1.0 - brier / baseline_brier) if baseline_brier else 0.0,
    }


def walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    min_train: int = MIN_TRAIN,
    refit_every: int = REFIT_EVERY,
    c: float = 0.1,
) -> Backtest:
    """Expanding-window backtest: never predict a day the model has seen."""
    labelled = labels.notna()
    features, labels = features.loc[labelled], labels.loc[labelled].astype(int)
    if len(features) <= min_train:
        raise ValueError(f"need more than {min_train} labelled rows, got {len(features)}")

    x_all, y_all = features.to_numpy(), labels.to_numpy()
    chunks: list[pd.Series] = []
    for start in range(min_train, len(features), refit_every):
        stop = min(start + refit_every, len(features))
        pipeline = make_pipeline(c).fit(x_all[:start], y_all[:start])
        prob = pipeline.predict_proba(x_all[start:stop])[:, 1]
        chunks.append(pd.Series(prob, index=features.index[start:stop]))

    probabilities = pd.concat(chunks)
    outcomes = labels.loc[probabilities.index]
    return Backtest(
        probabilities=probabilities,
        outcomes=outcomes,
        metrics=_metrics(probabilities.to_numpy(), outcomes.to_numpy()),
    )


@dataclass(frozen=True)
class Calibration:
    """Maps a raw probability onto the confidence the model has actually earned.

    The final model is fitted on every row it will ever see, so on a wide,
    collinear feature set it nearly separates its own training data and reports
    things like 0.9999 — a certainty the walk-forward record flatly
    contradicts. Two corrections, both derived from that record: the log-odds
    are clipped to the range the model ever reached out of sample, then Platt
    scaling maps them onto the frequencies actually realised there.
    """

    slope: float
    intercept: float
    lo: float
    hi: float

    def __call__(self, probabilities: np.ndarray) -> np.ndarray:
        logit = np.clip(_logit(np.asarray(probabilities, dtype=float)), self.lo, self.hi)
        return 1.0 / (1.0 + np.exp(-(self.slope * logit + self.intercept)))


def calibrator(backtest: Backtest) -> Calibration:
    logit = _logit(backtest.probabilities.to_numpy())
    platt = LogisticRegression(C=1.0, solver="lbfgs").fit(
        logit.reshape(-1, 1), backtest.outcomes.to_numpy()
    )
    return Calibration(
        slope=float(platt.coef_[0, 0]),
        intercept=float(platt.intercept_[0]),
        lo=float(logit.min()),
        hi=float(logit.max()),
    )


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(clipped / (1 - clipped))


def fit(features: pd.DataFrame, labels: pd.Series, c: float = 0.1) -> Pipeline:
    labelled = labels.notna()
    return make_pipeline(c).fit(
        features.loc[labelled].to_numpy(), labels.loc[labelled].astype(int).to_numpy()
    )


def coefficients(pipeline: Pipeline, columns: list[str]) -> pd.Series:
    """Standardised coefficients, i.e. log-odds impact of a one-sigma move."""
    weights = pipeline.named_steps["clf"].coef_[0]
    return pd.Series(weights, index=columns).sort_values(key=abs, ascending=False)
