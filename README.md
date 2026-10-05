# DengAI: Predicting Disease Spread

Coursework for [DrivenData's DengAI competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/):
predict weekly dengue cases in San Juan (Puerto Rico) and Iquitos (Peru) from climate data.

## Layout

```
DengAI_notebook.ipynb             THE notebook - imports everything from DengAI/
DengAI/                           the package: transformers, estimators, pipelines
  transformers.py                 DengueImputer, WeatherHistoryTransformer, cyclical_encoding
  selectors.py                    CatBoostFeatureSelector
  models/                         ProphetRegressor, SARIMAXRegressor, seasonal baselines
  pipelines.py                    make_city_pipeline - four branches
  evaluate.py                     chronological CV, grid search, holdout, submissions
  config.py data.py paths.py
configs/
  features.yaml                   column groups, lags/windows, per-city imputation
  models.yaml                     per-model params, grids, branch, artifact names
data/                             competition CSVs
experiments/                      side experiments that are not part of the main pipeline
presentation/                     slide decks and handouts (markdown + PDF)
models/                           8 fitted pipelines (.pkl) + best_params_models.json
submission.csv                    competition submission
run_full_search.py                rebuild every model from scratch (~2h)
archive/                          earlier exploratory work - nothing here is needed
```

Everything resolves paths from the repo root, so the notebook runs from anywhere.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .          # makes `import DengAI` work from any cwd
.venv/bin/python -m ipykernel install --user --name dengai-venv \
    --display-name "Python (DengAI .venv)"
```

**Run the notebook on the `Python (DengAI .venv)` kernel.** Its stored kernelspec points at
the system Python, which does not have catboost or prophet.

## How configuration works

One rule: **every mapping in the YAML is exactly the `__init__` kwargs of the object it
configures**, expanded with `**`. There is no schema layer and no validation framework — a
typo becomes a `TypeError` when the pipeline is built, which is the failure you want.

```python
from DengAI import load_config, build_pipelines
cfg = load_config()
pipelines = build_pipelines(cfg)      # {city: {model_name: Pipeline}}
```

To change the lag grid, edit `configs/features.yaml`. To add a model, add a block to
`configs/models.yaml` naming its estimator, branch, params and grid — no Python changes.

## The four pipeline branches

| Branch | Steps | Models |
|---|---|---|
| `tree` | imputer → history → cyclical → features → select → model | catboost, random_forest |
| `dense` | the tree steps, then impute + scale | ridge, lasso |
| `time_series` | imputer → history → ColumnTransformer(date + weather) → model | prophet, sarimax |
| `seasonal` | ColumnTransformer(year, weekofyear) → model | seasonal_median, shape_level |

`dense` exists because the tree branch leaves NaNs in the lag/rolling warm-up: trees tolerate
them, `RidgeCV` raises `ValueError: Input X contains NaN`. `seasonal` exists because the
calendar-only models need `weekofyear` and `year`, which `model_features` drops.

## Results (holdout MAE, 52-week chronological block)

| Model | San Juan | Iquitos |
|---|---|---|
| **catboost** | **15.02** | 6.27 |
| random_forest | 15.16 | 8.39 |
| **shape_level** | 20.72 | **3.39** |
| lasso | 21.16 | 4.50 |
| prophet | 21.78 | 7.16 |
| seasonal_median | 23.49 | 4.41 |
| ridge | 24.46 | 5.48 |
| sarimax | 24.56 | 4.61 |

The notable result is that **`shape_level` — which never looks at the weather — is the best
Iquitos model**, beating SARIMAX and CatBoost. Consistent with the exploratory finding that
Iquitos' climate carries almost no predictive signal.

## Three bugs found and fixed during the refactor

**1 · The notebook could not run with `SEARCH_JOBS=2`.** catboost's `select_features` ignores
`allow_writing_files=False` and does a bare `mkdir("catboost_info")`; with `n_jobs>1` two
joblib workers race and one dies with `FileExistsError`. Sequential calls are fine — only the
parallel case fails. `CatBoostFeatureSelector` now gives each fit its own `train_dir`.

**2 · `ShapeLevelRegressor` must read the `year` column, not `isocalendar().year`.** The two
disagree on 10 training and 5 test rows at Jan-1 boundaries, which would misassign those weeks
and bias the annual level. (`weekofyear` is safe to derive — 0 mismatches.)

**3 · The submission model choice was declared twice, inconsistently** — `iq: catboost` in one
cell and `iq: sarimax` in another. It is now stated once, in `configs/models.yaml`.

## Verification

The extracted classes were checked against the notebook originals before anything was deleted:

```
sj  DengueImputer             IDENTICAL  (936, 24)
sj  WeatherHistoryTransformer IDENTICAL  (52, 128), 104 generated cols
iq  DengueImputer             IDENTICAL  (520, 24)
iq  WeatherHistoryTransformer IDENTICAL  (52, 128), 104 generated cols
```

And the package reproduces the original results: **SARIMAX San Juan holdout 24.564 vs 24.564
recorded (exact)**, Iquitos CV 6.982 (exact), with identical CV fold boundaries. Prophet drifts
0.1–0.3 MAE because Stan is not bit-reproducible across versions.

Reference numbers from before the refactor are in `.baseline_backup/REFERENCE_MAE.md`.

## Rebuilding everything

```bash
.venv/bin/python scripts/run_full_search.py          # all 8 models, both cities (~2h)
```

Writes fitted pipelines to `models/`, results to `submissions/results_full_search.csv`, and a
submission CSV. The notebook does the same thing interactively.
