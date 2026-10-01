"""YAML configuration loading.

One rule keeps this simple: **every mapping under a transformer or estimator key is
exactly that object's ``__init__`` keyword arguments**, so it is consumed by ``**``
expansion with no translation layer. A typo in a YAML key therefore surfaces as a
``TypeError`` when the pipeline is built, which is the failure you want.
"""
from __future__ import annotations

import functools

import yaml

from .paths import CONFIG_DIR, check_layout


class Cfg(dict):
    """A ``dict`` with attribute access.

    Subclasses ``dict`` rather than using ``SimpleNamespace`` so that
    ``**cfg.weather_history`` still works -- a namespace is not a mapping.
    """

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key) from None
        return Cfg(value) if isinstance(value, dict) else value

    def __dir__(self):
        return list(self.keys()) + list(super().__dir__())


def _load_yaml(name: str) -> Cfg:
    path = CONFIG_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return Cfg(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


@functools.lru_cache(maxsize=None)
def load_config(features: str = "features.yaml", models: str = "models.yaml") -> Cfg:
    """Load and cache both config files.

    Note the result is cached and mutable: never mutate it inside package code.
    """
    check_layout()
    return Cfg(features=_load_yaml(features), models=_load_yaml(models))
