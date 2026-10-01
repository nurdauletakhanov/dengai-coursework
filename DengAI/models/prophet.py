"""Prophet wrapped as an sklearn regressor. Ported verbatim from the main notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ProphetRegressor(RegressorMixin, BaseEstimator):
    """Prophet with every non-date column added as an external regressor.

    Standardisation is left to the pipeline's ``StandardScaler``, hence
    ``add_regressor(..., standardize=False)``. ``__getstate__``/``__setstate__``
    serialise the Stan model through Prophet's JSON helpers so the whole
    ``Pipeline`` stays picklable.
    """

    def __init__(
        self,
        yearly_seasonality=5,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode="additive",
        growth="linear",
        random_state=2022,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.growth = growth
        self.random_state = random_state

    def _frame(self, X):
        frame = X.rename(columns={"week_start_date": "ds"}).copy()
        frame["ds"] = pd.to_datetime(frame["ds"])
        return frame

    def fit(self, X, y):
        from prophet import Prophet

        self.regressor_columns_ = X.columns.drop("week_start_date").tolist()
        self.model_ = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            growth=self.growth,
            uncertainty_samples=0,
        )
        for col in self.regressor_columns_:
            self.model_.add_regressor(col, standardize=False)
        frame = self._frame(X)
        frame["y"] = np.asarray(y, dtype=float)
        self.model_.fit(frame, seed=self.random_state)
        return self

    def predict(self, X):
        predictions = self.model_.predict(self._frame(X))["yhat"].to_numpy()
        return np.maximum(predictions, 0)

    def __getstate__(self):
        from prophet.serialize import model_to_json

        state = self.__dict__.copy()
        if "model_" in state:
            state["model_json_"] = model_to_json(state.pop("model_"))
        return state

    def __setstate__(self, state):
        from prophet.serialize import model_from_json

        self.__dict__.update(state)
        if "model_json_" in state:
            self.model_ = model_from_json(self.__dict__.pop("model_json_"))
