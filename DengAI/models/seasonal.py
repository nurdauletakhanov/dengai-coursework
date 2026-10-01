"""Calendar-only baselines.

Neither model looks at the weather. They exist because the analysis found that climate
sets *when* cases rise within a year but explains almost nothing about *how many* --
so a seasonal profile plus a typical annual level is a strong, honest baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from ..transformers import circular_smooth


def _weeks(X, week_column, date_column="week_start_date"):
    """Week-of-year as a 0..n-1 indexed Series.

    Prefers the ``weekofyear`` column; falls back to the date. Verified equivalent on
    this dataset: 0 mismatches across all 1456 train and 416 test rows.
    """
    if week_column in X.columns:
        weeks = X[week_column]
    else:
        weeks = pd.to_datetime(X[date_column]).dt.isocalendar().week
    return pd.Series(np.asarray(weeks, dtype=int), index=np.arange(len(X)))


class SeasonalMedianRegressor(RegressorMixin, BaseEstimator):
    """Predict each week with a smoothed per-week-of-year statistic of past cases."""

    def __init__(self, statistic="median", smooth_window=5, week_column="weekofyear"):
        self.statistic = statistic
        self.smooth_window = smooth_window
        self.week_column = week_column

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        y = np.asarray(y, dtype=float)
        profile = pd.Series(y).groupby(_weeks(X, self.week_column)).agg(self.statistic)
        self.profile_ = circular_smooth(profile, self.smooth_window)
        self.fallback_ = float(np.median(y))
        return self

    def predict(self, X):
        weeks = _weeks(X, self.week_column)
        values = weeks.map(self.profile_).fillna(self.fallback_).to_numpy()
        return np.maximum(values, 0.0)


class ShapeLevelRegressor(RegressorMixin, BaseEstimator):
    """Seasonal shape (normalised to sum 1) multiplied by a typical annual total.

    The decomposition the analysis argued for: the *shape* of a dengue year is
    reproducible from the calendar, while its *level* is driven by which serotype is
    circulating -- information absent from this dataset -- so the level is estimated
    as the median annual total rather than predicted.

    Uses the ``year`` COLUMN, never ``isocalendar().year``: the two disagree on 10
    training and 5 test rows at Jan-1 boundaries, which would misassign those weeks to
    the adjacent year and bias the level.
    """

    def __init__(
        self,
        shape_statistic="mean",
        smooth_window=5,
        min_weeks_per_year=45,
        level_statistic="median",
        week_column="weekofyear",
        year_column="year",
    ):
        self.shape_statistic = shape_statistic
        self.smooth_window = smooth_window
        self.min_weeks_per_year = min_weeks_per_year
        self.level_statistic = level_statistic
        self.week_column = week_column
        self.year_column = year_column

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        y = pd.Series(np.asarray(y, dtype=float), index=np.arange(len(X)))
        weeks = _weeks(X, self.week_column)

        shape = y.groupby(weeks).agg(self.shape_statistic)
        shape = circular_smooth(shape, self.smooth_window)
        self.shape_ = shape / shape.sum()

        if self.year_column in X.columns:
            years = pd.Series(
                np.asarray(X[self.year_column], dtype=int), index=y.index
            )
            totals = y.groupby(years).sum()
            # Partial years at the edges of a training window drag the median down.
            complete = years.value_counts().reindex(totals.index)
            totals = totals[complete >= self.min_weeks_per_year]
            self.level_ = (
                float(getattr(totals, self.level_statistic)())
                if len(totals)
                else float(y.sum() / max(years.nunique(), 1))
            )
        else:  # no year column: fall back to annualising the mean weekly rate
            self.level_ = float(y.mean() * 52)

        self.fallback_ = 1.0 / 52.0
        return self

    def predict(self, X):
        weeks = _weeks(X, self.week_column)
        share = weeks.map(self.shape_).fillna(self.fallback_).to_numpy()
        return np.maximum(share * self.level_, 0.0)
