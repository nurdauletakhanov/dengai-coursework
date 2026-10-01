"""Estimators. Heavy third-party imports happen lazily, inside ``fit``."""
from __future__ import annotations

import importlib

from .sarimax import SARIMAXRegressor
from .seasonal import SeasonalMedianRegressor, ShapeLevelRegressor

__all__ = [
    "SARIMAXRegressor",
    "SeasonalMedianRegressor",
    "ShapeLevelRegressor",
    "ProphetRegressor",
    "build_model",
]


def __getattr__(name):
    # ProphetRegressor is lazy: importing it must not require prophet to be installed.
    if name == "ProphetRegressor":
        from .prophet import ProphetRegressor

        return ProphetRegressor
    raise AttributeError(name)


def _resolve(dotted: str):
    """Import ``package.module.Name`` and return the attribute."""
    module, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def build_model(name, cfg=None, seed=None):
    """Construct the estimator named in ``configs/models.yaml``."""
    if cfg is None:
        from ..config import load_config

        cfg = load_config()
    spec = cfg.models.models[name]
    params = dict(spec.get("params", {}))
    if seed is not None and "seed_param" in spec:
        params[spec["seed_param"]] = seed
    return _resolve(spec["estimator"])(**params)
