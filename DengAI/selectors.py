"""CatBoost-based feature selection.

``catboost`` is imported inside ``fit`` so that ``import DengAI`` works on a venv
where catboost is not installed.
"""
from __future__ import annotations

import tempfile

from sklearn.base import BaseEstimator, TransformerMixin


class CatBoostFeatureSelector(TransformerMixin, BaseEstimator):
    """Select ``n_features`` columns by recursive SHAP-value elimination.

    Two differences from the notebook original, both deliberate:

    1. ``catboost_params`` is a constructor argument instead of a module-level global.
       It resolves to the config default inside ``fit`` (never in ``__init__``, which
       sklearn's clone contract requires to store arguments unmodified).
    2. ``fit`` passes ``train_dir`` pointing at a per-fit temporary directory.
       ``select_features`` ignores ``allow_writing_files=False`` and does a bare
       ``mkdir("catboost_info")`` in the cwd; under ``n_jobs>1`` two workers race and
       one dies with ``FileExistsError``. Verified: sequential calls are fine, only
       the parallel case fails.
    """

    def __init__(self, n_features=15, catboost_params=None, steps=3):
        self.n_features = n_features
        self.catboost_params = catboost_params
        self.steps = steps

    def _params(self):
        if self.catboost_params is not None:
            return dict(self.catboost_params)
        from .config import load_config

        return dict(load_config().models.models.catboost.params)

    def fit(self, X, y):
        from catboost import CatBoostRegressor, EFeaturesSelectionAlgorithm, Pool

        with tempfile.TemporaryDirectory() as train_dir:
            model = CatBoostRegressor(**self._params(), train_dir=train_dir)
            selection = model.select_features(
                Pool(X, y),
                features_for_select=list(X.columns),
                num_features_to_select=self.n_features,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                steps=self.steps,
                train_final_model=False,
                verbose=False,
                log_cout=lambda _: None,
            )
        self.selected_features_ = selection["selected_features_names"]
        return self

    def transform(self, X):
        return X[self.selected_features_]
