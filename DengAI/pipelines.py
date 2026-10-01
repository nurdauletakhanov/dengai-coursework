"""Pipeline construction -- the single place YAML config meets sklearn."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import load_config
from .models import build_model
from .selectors import CatBoostFeatureSelector
from .transformers import (
    DengueImputer,
    WeatherHistoryTransformer,
    cyclical_encoding,
    model_features,
)

#: ``"model"`` is always the final step, so grid keys stay ``model__*``.
BRANCHES = ("tree", "dense", "time_series", "seasonal")


def _weather_history(cfg):
    wh = dict(cfg.features.weather_history)
    # lag_columns is derived, so the two lists can never drift apart.
    wh["lag_columns"] = list(wh["mean_columns"]) + list(wh["sum_columns"])
    return WeatherHistoryTransformer(**wh)


def make_city_pipeline(model, city="sj", branch="tree", cfg=None):
    """Build the pipeline for one city and one model.

    Four branches:

    ``tree``        imputer -> history -> cyclical -> features -> select -> model
    ``dense``       the tree steps, then impute + scale (Ridge/Lasso reject NaN)
    ``time_series`` imputer -> history -> ColumnTransformer(date + scaled weather) -> model
    ``seasonal``    ColumnTransformer(year, weekofyear) -> model   (calendar only)
    """
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")
    cfg = cfg or load_config()
    f = cfg.features

    if branch == "seasonal":
        # No weather at all, so no imputer and no history: just the calendar columns.
        calendar = ColumnTransformer(
            [("calendar", "passthrough", list(f.branches.calendar_columns))],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        return Pipeline([("features", calendar), ("model", model)])

    steps = [
        (
            "imputer",
            DengueImputer(
                seasonal_columns=tuple(
                    f.imputation.seasonal_columns_by_city.get(city, ())
                ),
                copy=True,
            ),
        ),
        ("weather_history", _weather_history(cfg)),
    ]

    if branch == "time_series":
        weather = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        steps.append(
            (
                "features",
                ColumnTransformer(
                    [
                        ("date", "passthrough", ["week_start_date"]),
                        ("weather", weather, list(f.columns.time_series)),
                    ],
                    verbose_feature_names_out=False,
                ).set_output(transform="pandas"),
            )
        )
    else:  # tree, dense
        drop = list(f.model_features.drop)
        steps += [
            ("cyclical", FunctionTransformer(cyclical_encoding, kw_args={"copy": True})),
            ("features", FunctionTransformer(model_features, kw_args={"drop": drop})),
            ("select_features", CatBoostFeatureSelector(**f.feature_selection)),
        ]
        if branch == "dense":
            steps += [
                ("fill", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]

    return Pipeline(steps + [("model", model)])


def build_pipelines(cfg=None, names=None, cities=("sj", "iq")):
    """Return ``{city: {model_name: Pipeline}}`` for the requested models."""
    cfg = cfg or load_config()
    names = list(names or cfg.models.models.keys())
    seed = cfg.features.random_seed
    out = {}
    for city in cities:
        out[city] = {
            name: make_city_pipeline(
                build_model(name, cfg, seed=seed),
                city=city,
                branch=cfg.models.models[name].get("branch", "tree"),
                cfg=cfg,
            )
            for name in names
        }
    return out
