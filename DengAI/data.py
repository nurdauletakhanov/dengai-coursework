"""Data loading and city splitting.

Paths resolve from the repo root, not the cwd, so this works identically from
``Artem_notebook/``, from the repo root, and from a plain script.
"""
from __future__ import annotations

import pandas as pd

from .paths import DATA_DIR


def load_raw():
    """Return ``(train_features, train_labels, test_features)``."""
    return (
        pd.read_csv(DATA_DIR / "dengue_features_train.csv"),
        pd.read_csv(DATA_DIR / "dengue_labels_train.csv"),
        pd.read_csv(DATA_DIR / "dengue_features_test.csv"),
    )


def split_cities(train_features, train_labels, test_features, cities=("sj", "iq")):
    """Split every frame by city, preserving row order.

    Returns ``{city: {"X_train":..., "y_train":..., "X_test":...}}``.
    """
    out = {}
    for city in cities:
        mask_tr = train_features["city"] == city
        out[city] = {
            "X_train": train_features[mask_tr].reset_index(drop=True),
            "y_train": train_labels[train_labels["city"] == city][
                "total_cases"
            ].reset_index(drop=True),
            "X_test": test_features[test_features["city"] == city].reset_index(
                drop=True
            ),
        }
    return out
