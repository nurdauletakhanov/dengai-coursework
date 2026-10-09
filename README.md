# DengAI: Predicting Disease Spread

Coursework for [DrivenData's DengAI competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/):
predict weekly dengue cases in San Juan (Puerto Rico) and Iquitos (Peru) from climate data.

## Layout

```
DengAI_notebook.ipynb             THE notebook - imports everything from DengAI/
DengAI/                           the package: transformers, estimators, pipelines
  transformers.py                 DengueImputer, WeatherHistoryTransformer, cyclical_encoding
  selectors.py                    CatBoostFeatureSelector, ridge_rfe
  plots.py                        presentation figures: city patterns, lag evidence, folds, pipeline, missingness
  models/                         ProphetRegressor, SARIMAXRegressor, seasonal baselines
  pipelines.py                    make_city_pipeline - five branches
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
run_full_search.py                rebuild every model from scratch (~8h)
archive/                          earlier exploratory work
```

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

## The five pipeline branches

| Branch | Steps | Models |
|---|---|---|
| `tree` | imputer → history → cyclical → features → select → model | catboost, random_forest |
| `dense` | the tree steps, then impute + scale | ridge, lasso |
| `rfe` | imputer → history → cyclical → features → impute + scale → Ridge-RFE → model | xgboost |
| `time_series` | imputer → history → ColumnTransformer(date + weather) → model | prophet, sarimax |
| `seasonal` | ColumnTransformer(year, weekofyear) → model | seasonal_median, shape_level |

`dense` exists because the tree branch leaves NaNs in the lag/rolling warm-up: trees tolerate
them, `RidgeCV` raises `ValueError: Input X contains NaN`. `seasonal` exists because the
calendar-only models need `weekofyear` and `year`, which `model_features` drops. `rfe` exists
because Ridge-RFE, the selector that won the feature-selection experiment, cannot see NaN,
so it must come after imputing and scaling rather than before, as the CatBoost selector does.

## Results (holdout MAE, 52-week chronological block)

| Model | San Juan | Iquitos |
|---|---|---|
| **xgboost** | **13.94** | 4.17 |
| catboost | 15.02 | 6.27 |
| random_forest | 15.16 | 8.39 |
| **shape_level** | 20.72 | **3.39** |
| lasso | 21.42 | 4.44 |
| prophet | 21.78 | 7.16 |
| seasonal_median | 23.49 | 4.41 |
| ridge | 24.46 | 5.48 |
| sarimax | 24.56 | 4.61 |

The notable result is that **`shape_level` — which never looks at the weather — is the best
Iquitos model**, beating SARIMAX and CatBoost. Consistent with the exploratory finding that
Iquitos' climate carries almost no predictive signal.
