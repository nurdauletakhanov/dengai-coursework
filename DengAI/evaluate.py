"""Chronological evaluation: splitting, grid search, holdout scoring, submissions.

Every split here is chronological. That is not a stylistic choice: ``SARIMAXRegressor``
raises unless the block it predicts starts strictly after its training data, and the
whole dataset is a time series where a shuffled split would leak the answer.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .config import load_config
from .paths import ROOT

CITY_NAMES = {"sj": "San Juan", "iq": "Iquitos"}


def missing_report(df):
    """Per-column missing counts and percentages, worst first."""
    n = df.isna().sum()
    out = pd.DataFrame({"missing": n, "percent": (100 * n / len(df)).round(2)})
    return out[out["missing"] > 0].sort_values("missing", ascending=False)


KEY = ("city", "year", "weekofyear")


def duplicate_report(df, labels=None, key=KEY):
    """Duplicate rows, duplicate keys, feature/label mismatches and identical columns.

    One row per check with a count and, where useful, the offending names. A clean
    dataset shows zeros everywhere except ``identical columns``, which flags
    ``reanalysis_sat_precip_amt_mm`` as a copy of ``precipitation_amt_mm``.
    """
    key = list(key)
    rows = {
        "duplicated rows": (int(df.duplicated().sum()), ""),
        f"duplicated keys {key}": (int(df.duplicated(key).sum()), ""),
    }
    if "week_start_date" in df.columns:
        n = int(df.duplicated(["city", "week_start_date"]).sum())
        rows["duplicated (city, week_start_date)"] = (n, "")
    if labels is not None:
        merged = df[key].merge(labels[key], how="outer", indicator=True)
        rows["feature rows without a label"] = (int((merged["_merge"] == "left_only").sum()), "")
        rows["label rows without features"] = (int((merged["_merge"] == "right_only").sum()), "")
    numeric = df.select_dtypes("number")
    cols = list(numeric.columns)
    pairs = [
        f"{a} == {b}"
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
        if numeric[a].equals(numeric[b])
    ]
    rows["identical columns"] = (len(pairs), "; ".join(pairs))
    return pd.DataFrame(rows, index=["count", "detail"]).T


def make_cv(cfg=None):
    cfg = cfg or load_config()
    ev = cfg.models.evaluation
    return TimeSeriesSplit(n_splits=ev.cv_splits, test_size=ev.cv_weeks)


def city_data(cities, cfg=None):
    """Carve a chronological holdout off the tail of each city's training series."""
    cfg = cfg or load_config()
    h = cfg.models.evaluation.holdout_weeks
    out = {}
    for city, frames in cities.items():
        X, y = frames["X_train"], frames["y_train"]
        out[city] = {
            "X_dev": X.iloc[:-h],
            "y_dev": y.iloc[:-h],
            "X_holdout": X.iloc[-h:],
            "y_holdout": y.iloc[-h:],
            "X_train": X,
            "y_train": y,
            "X_test": frames["X_test"],
        }
    return out


def fold_table(data, cv):
    """Show each CV fold's train end and validation span -- a leakage sanity check."""
    rows = []
    for city, d in data.items():
        dates = pd.to_datetime(d["X_dev"]["week_start_date"]).reset_index(drop=True)
        for i, (tr, va) in enumerate(cv.split(d["X_dev"]), start=1):
            rows.append(
                {
                    "City": CITY_NAMES.get(city, city),
                    "Fold": i,
                    "Train ends": dates.iloc[tr[-1]].date(),
                    "Val starts": dates.iloc[va[0]].date(),
                    "Val ends": dates.iloc[va[-1]].date(),
                }
            )
    return pd.DataFrame(rows)


def _error_score(spec):
    value = spec.get("error_score", "raise")
    return np.nan if value in ("nan", "NaN", None) else value


def param_grids(cfg=None, names=None):
    cfg = cfg or load_config()
    names = names or list(cfg.models.models.keys())
    return {n: cfg.models.models[n].get("grid", {}) for n in names}


def run_searches(pipelines, data, cfg=None, names=None, verbose=True):
    """Grid-search every (city, model) pair. Returns ``(searches, best, cv_results)``."""
    cfg = cfg or load_config()
    ev = cfg.models.evaluation
    cv = make_cv(cfg)
    names = names or sorted({n for city in pipelines.values() for n in city})
    searches, best, rows = {}, {}, []
    for city, by_model in pipelines.items():
        searches[city], best[city] = {}, {}
        for name in names:
            if name not in by_model:
                continue
            spec = cfg.models.models[name]
            search = GridSearchCV(
                by_model[name],
                spec.get("grid", {}),
                cv=cv,
                scoring=ev.scoring,
                n_jobs=ev.search_jobs,
                error_score=_error_score(spec),
            )
            search.fit(data[city]["X_dev"], data[city]["y_dev"])
            searches[city][name] = search
            best[city][name] = search.best_estimator_
            rows.append(
                {
                    "City": CITY_NAMES.get(city, city),
                    "Model": name,
                    "CV MAE": -search.best_score_,
                    "Best params": search.best_params_,
                }
            )
            if verbose:
                print(f"  {CITY_NAMES.get(city, city):9s} {name:16s} CV MAE {-search.best_score_:7.3f}")
    return searches, best, pd.DataFrame(rows)


def holdout_table(best, data, verbose=True):
    """Score each dev-fitted pipeline on its city's untouched holdout block."""
    rows, preds = [], {}
    for city, by_model in best.items():
        preds[city] = {}
        for name, pipe in by_model.items():
            try:
                p = pipe.predict(data[city]["X_holdout"])
                mae = mean_absolute_error(data[city]["y_holdout"], p)
            except Exception as exc:  # SARIMAX refuses non-future blocks, etc.
                p, mae = None, np.nan
                if verbose:
                    print(f"  {city}/{name}: {type(exc).__name__}: {exc}")
            preds[city][name] = p
            rows.append(
                {"City": CITY_NAMES.get(city, city), "Model": name, "Holdout MAE": mae}
            )
    return pd.DataFrame(rows), preds


def refit_final(best, data):
    """Refit the chosen pipelines on the full training series, for test prediction."""
    return {
        city: {
            name: clone(pipe).fit(data[city]["X_train"], data[city]["y_train"])
            for name, pipe in by_model.items()
        }
        for city, by_model in best.items()
    }


def make_submission(final, data, cfg=None, path=None):
    """Build the DrivenData submission using the per-city model named in config."""
    cfg = cfg or load_config()
    choice = dict(cfg.models.submission.model_by_city)
    frames = []
    for city, name in choice.items():
        X = data[city]["X_test"]
        pred = np.rint(final[city][name].predict(X)).clip(0).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "city": X["city"].to_numpy(),
                    "year": X["year"].to_numpy(),
                    "weekofyear": X["weekofyear"].to_numpy(),
                    "total_cases": pred,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    if path:
        out.to_csv(path, index=False)
    return out


def _artifact_path(name, cfg):
    d = ROOT / cfg.models.artifacts.dir
    d.mkdir(parents=True, exist_ok=True)
    return d / cfg.models.artifacts.files[name]


def save_pipelines(final, cfg=None, names=None):
    """Persist ``{city: pipeline}`` per model, with a version stamp beside it.

    Classes now live in the installed package, so cloudpickle stores them **by
    reference**. That is deliberate: a by-value pickle would silently keep running
    old code after the package changed, whereas a by-reference one fails loudly.
    """
    import cloudpickle

    cfg = cfg or load_config()
    import sklearn

    from . import __version__

    meta = {
        "DengAI": __version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "fitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    written = []
    for name in names or sorted({n for c in final.values() for n in c}):
        path = _artifact_path(name, cfg)
        payload = {"meta": meta, "pipelines": {c: final[c][name] for c in final}}
        with path.open("wb") as fh:
            cloudpickle.dump(payload, fh)
        written.append(path)
    return written


def load_pipelines(name, cfg=None, warn=True):
    """Load a saved artifact, warning if it was fitted under different libraries."""
    import cloudpickle
    import sklearn

    cfg = cfg or load_config()
    path = _artifact_path(name, cfg)
    with path.open("rb") as fh:
        payload = cloudpickle.load(fh)
    if isinstance(payload, dict) and "pipelines" in payload:
        meta = payload.get("meta", {})
        if warn and meta.get("sklearn") != sklearn.__version__:
            print(
                f"  note: {path.name} was fitted with sklearn {meta.get('sklearn')}, "
                f"running {sklearn.__version__}"
            )
        return payload["pipelines"]
    return payload  # legacy artifact: a bare {city: pipeline} dict
