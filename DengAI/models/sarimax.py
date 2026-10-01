"""SARIMAX wrapped as an sklearn regressor. Ported verbatim from the main notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAXRegressor(RegressorMixin, BaseEstimator):
    """SARIMAX with Fourier annual seasonality built into its exogenous block.

    Two deliberate hard constraints, both inherited from the notebook: ``fit`` raises
    if the optimiser does not converge, and ``predict`` refuses any block that is not
    strictly after the training data. Together they are why evaluation must stay
    chronological -- no shuffling, no ``cross_val_predict``.

    ``order`` and ``seasonal_order`` are coerced to tuples inside ``fit`` so that YAML
    lists work; ``__init__`` stores them unmodified, as sklearn's clone contract requires.
    """

    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        fourier_order=3,
        trend="c",
        maxiter=200,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fourier_order = fourier_order
        self.trend = trend
        self.maxiter = maxiter

    def _exog(self, X):
        dates = pd.to_datetime(X["week_start_date"])
        days_in_year = 365 + dates.dt.is_leap_year.astype(int)
        angle = 2 * np.pi * (dates.dt.dayofyear - 1) / days_in_year
        fourier = {}
        for k in range(1, self.fourier_order + 1):
            fourier[f"annual_sin_{k}"] = np.sin(k * angle).to_numpy()
            fourier[f"annual_cos_{k}"] = np.cos(k * angle).to_numpy()
        weather = X.drop(columns="week_start_date").reset_index(drop=True)
        return pd.concat([weather, pd.DataFrame(fourier, index=weather.index)], axis=1)

    def fit(self, X, y):
        self.train_end_ = pd.to_datetime(X["week_start_date"]).iloc[-1]
        self.model_ = SARIMAX(
            endog=np.asarray(y, dtype=float),
            exog=self._exog(X),
            order=tuple(self.order),
            seasonal_order=tuple(self.seasonal_order),
            trend=self.trend,
        ).fit(disp=False, maxiter=self.maxiter)
        self.converged_ = bool(self.model_.mle_retvals["converged"])
        if not self.converged_:
            raise RuntimeError(
                "SARIMAX did not converge. Increase model__maxiter or simplify "
                "the order/weather features before comparing scores."
            )
        return self

    def predict(self, X):
        if pd.to_datetime(X["week_start_date"]).iloc[0] <= self.train_end_:
            raise ValueError("SARIMAX needs a future block after its training data.")
        forecast = self.model_.get_forecast(steps=len(X), exog=self._exog(X))
        return np.maximum(np.asarray(forecast.predicted_mean), 0)
