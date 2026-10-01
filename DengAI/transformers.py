"""Feature transformers.

``DengueImputer`` and ``WeatherHistoryTransformer`` are ported verbatim from the main
notebook; only the documented fixes noted in each docstring differ. Nothing here
imports a heavy third-party library, so this module always imports cleanly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DengueImputer(TransformerMixin, BaseEstimator):
    """Fill missing weather values.

    Seasonal columns (San Juan's NDVI, whose north-east pixel falls over the Atlantic)
    are filled from week-of-year training medians, then the overall training median.
    Everything else is linearly interpolated.

    ``copy`` defaults to ``True`` here, unlike the notebook original -- in-place
    mutation of the caller's frame is surprising, and every pipeline call site
    already passed ``copy=True``.
    """

    def __init__(self, seasonal_columns=(), copy=True):
        self.seasonal_columns = seasonal_columns
        self.copy = copy

    def fit(self, X, y=None):
        excluded = ["city", "year", "weekofyear", "week_start_date", "total_cases"]
        self.feature_columns_ = [c for c in X.columns if c not in excluded]
        self.interpolation_columns_ = [
            c for c in self.feature_columns_ if c not in self.seasonal_columns
        ]
        seasonal = list(self.seasonal_columns)
        self.week_medians_ = X.groupby("weekofyear")[seasonal].median()
        self.overall_medians_ = X[seasonal].median()
        return self

    def transform(self, X):
        result = X.copy() if self.copy else X

        for col in self.seasonal_columns:
            seasonal_values = result["weekofyear"].map(self.week_medians_[col])
            result[col] = result[col].fillna(seasonal_values).fillna(
                self.overall_medians_[col]
            )

        result[self.interpolation_columns_] = result[
            self.interpolation_columns_
        ].interpolate(method="linear", limit_area="inside")
        return result


class WeatherHistoryTransformer(TransformerMixin, BaseEstimator):
    """Lag and rolling-window features, without leaking across a fold boundary.

    ``fit`` memoises the tail of the training fold; ``transform`` prepends only the
    rows strictly earlier than the incoming block, so lags at the start of a
    validation or test block are computed from real past data rather than NaN.

    Generated column names: ``{col}_lag_{n}w``, ``{col}_sum_{n}w``, ``{col}_mean_{n}w``.
    """

    def __init__(
        self,
        lag_columns,
        mean_columns,
        sum_columns,
        lags=(1, 2, 4, 8, 10, 20, 26),
        windows=(4, 8, 10, 12, 16, 26),
    ):
        self.lag_columns = lag_columns
        self.mean_columns = mean_columns
        self.sum_columns = sum_columns
        self.lags = lags
        self.windows = windows

    def fit(self, X, y=None):
        history_length = max(max(self.lags), max(self.windows))
        self.history_ = X.tail(history_length).copy()
        return self

    def transform(self, X):
        history = self.history_.loc[
            self.history_["week_start_date"] < X["week_start_date"].iloc[0]
        ]
        result = pd.concat([history, X])
        features = {}

        for col in self.lag_columns:
            for lag in self.lags:
                features[f"{col}_lag_{lag}w"] = result[col].shift(lag)

        for window in self.windows:
            for col in self.sum_columns:
                features[f"{col}_sum_{window}w"] = (
                    result[col].shift(1).rolling(window).sum()
                )
            for col in self.mean_columns:
                features[f"{col}_mean_{window}w"] = (
                    result[col].shift(1).rolling(window).mean()
                )

        features = pd.DataFrame(features, index=result.index)
        result = pd.concat([result, features], axis=1)
        return result.iloc[-len(X):].copy()


def cyclical_encoding(df, copy=False):
    """Add ``annual_sin`` / ``annual_cos`` from the week-start date.

    Week 52 and week 1 are adjacent in the calendar but 51 apart as integers; placing
    the year on a circle removes that false discontinuity.

    ``copy=False`` reproduces the notebook's in-place behaviour (used eagerly on the
    EDA frames); pipelines pass ``copy=True``.
    """
    if copy:
        df = df.copy()
    dates = pd.to_datetime(df["week_start_date"])
    days_in_year = 365 + dates.dt.is_leap_year.astype(int)
    angle = 2 * np.pi * (dates.dt.dayofyear - 1) / days_in_year
    df["annual_sin"] = np.sin(angle)
    df["annual_cos"] = np.cos(angle)
    return df


def model_features(df, drop=("city", "week_start_date", "weekofyear")):
    """Drop identifier and duplicated columns before the estimator.

    ``drop`` is an explicit argument rather than a module-level global, so the
    function carries no hidden dependency on notebook state.
    """
    return df.drop(columns=[c for c in drop if c in df.columns])


def circular_smooth(profile, window, n_weeks=53):
    """Smooth a week-of-year profile across the year boundary.

    Tiles the profile three times so the rolling window wraps December into January,
    then returns the middle copy. Replaces five near-identical copies of this idiom
    that previously lived in the analysis scripts.
    """
    profile = profile.reindex(range(1, n_weeks + 1)).interpolate().bfill().ffill()
    tiled = pd.concat([profile] * 3).rolling(window, center=True, min_periods=1).mean()
    return pd.Series(
        tiled.to_numpy()[n_weeks : 2 * n_weeks], index=range(1, n_weeks + 1)
    )
