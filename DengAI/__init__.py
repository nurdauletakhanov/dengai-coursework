"""DengAI -- pipeline components for the DrivenData dengue competition.

Typical use from a notebook::

    from DengAI import load_config, load_raw, split_cities, build_pipelines
    cfg = load_config()
    pipelines = build_pipelines(cfg)
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import Cfg, load_config
from .data import load_raw, split_cities
from .pipelines import build_pipelines, make_city_pipeline
from .selectors import CatBoostFeatureSelector
from .transformers import (
    DengueImputer,
    WeatherHistoryTransformer,
    circular_smooth,
    cyclical_encoding,
    model_features,
)

__all__ = [
    "Cfg",
    "load_config",
    "load_raw",
    "split_cities",
    "build_pipelines",
    "make_city_pipeline",
    "DengueImputer",
    "WeatherHistoryTransformer",
    "CatBoostFeatureSelector",
    "cyclical_encoding",
    "model_features",
    "circular_smooth",
    "set_seed",
]


def set_seed(seed=None):
    """Seed ``random`` and ``numpy``. Returns the seed used."""
    import random

    import numpy as np

    if seed is None:
        seed = load_config().features.random_seed
    random.seed(seed)
    np.random.seed(seed)
    return seed
