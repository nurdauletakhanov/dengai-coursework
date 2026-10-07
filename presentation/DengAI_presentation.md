---
title: "DengAI --- Predicting Disease Spread"
subtitle: "Metrics and Models"
date: "October 2026"
toc: true
toc-depth: 1
---
# Metrics

## The competition metric is MAE

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}\bigl|\,\hat{y}_i - y_i\,\bigr|$$

The average number of cases we are off by, per week.

- One number over all **416 test weeks** (260 San Juan + 156 Iquitos)
- Lower is better; a perfect model scores 0
- **San Juan dominates**: 63% of the weeks and about four times the cases

## The target is skewed

![Weekly cases: most weeks are quiet, a few outbreak weeks form a long tail](figures/target_skew.png){width=70%}

San Juan: **mean 34, median 19**, 73% of weeks below the mean.
Skewness **4.5** (San Juan) and **4.0** (Iquitos).

::: notes
Mean cases per week are 34.2 in San Juan and 7.6 in Iquitos, but the median week has only
19 and 5; the maxima are 461 and 116. In both cities the worst 10% of weeks hold 43% of all
cases.

Skewness is a number for this lopsidedness: 0 means symmetric, like a bell curve; above 1
already counts as strongly skewed. Ours is 4.5 and 4.0.

Everyday analogy: income. A few billionaires raise the *average* income, but the *median*
still describes the typical person. For us this means that the rare outbreak weeks would
dominate any metric that punishes big errors heavily.
:::

## Why MAE and not RMSE

- RMSE squares errors: one 461-case week counts like **~500 ordinary weeks off by one**
- RMSE is dragged into fitting a few outbreaks; **MAE weights every week equally**
- MAE is minimised by the **median**, RMSE by the **mean** --- for skewed counts the
  median is the robust target
- Consequence: good models here are *conservative*

::: notes
Under RMSE the model concentrates on the handful of epidemic weeks and pays little
attention to the hundreds of ordinary ones. MAE treats a miss of 10 in a quiet week exactly
like a miss of 10 in an outbreak week.

Statistically, the prediction that minimises expected absolute error is the conditional
median; squared error is minimised by the conditional mean, which for a right-skewed
distribution sits far above the typical week. This also explains a behaviour we see later:
the models that do well on this metric predict the quiet-week level and rarely chase peaks.
:::

# Models

## The pipeline has four branches

`make_city_pipeline` builds one of four shapes, because model families need different
preprocessing:

| Branch | Steps | Models |
|----------------|------------------------------|-----------------|
| `tree` | impute → history → cyclical → select → model | CatBoost, RandomForest |
| `dense` | tree steps **+ impute + scale** | Ridge, Lasso |
| `time_series` | impute → history → date + scaled weather | Prophet, SARIMAX |
| `seasonal` | calendar columns only | Seasonal median, Shape×level |

## Why four branches

**`dense`** --- the lag/rolling warm-up leaves **NaNs** in the first 26 weeks

- Tree models handle NaN natively; `RidgeCV` raises `ValueError`
- So linear models get `SimpleImputer(median)` + `StandardScaler`

**`seasonal`** --- the calendar baselines need `year` and `weekofyear`, which the tree
branch drops

- They **never see the weather**: the baseline any weather model must beat

::: notes
Scaling matters for the linear models too: Ridge and Lasso penalise coefficient sizes and
are not scale-invariant, so unscaled features would be penalised unevenly.

The tree branch drops `city`, `week_start_date` and `weekofyear` before the model. Rather
than weaken that branch, the two calendar models get their own: a `ColumnTransformer`
passing through `year` and `weekofyear` and nothing else.
:::

## The nine models

\scriptsize

| Model | Family | Core assumption |
|----------------|--------------------|------------------------------------|
| CatBoost | gradient boosting | non-linear effects of lagged weather |
| XGBoost | gradient boosting | as CatBoost, on RFE-selected features |
| RandomForest | bagged trees | same, without boosting |
| Ridge | linear + L2 | smooth linear response, all features shrunk |
| Lasso | linear + L1 | linear, but most features are irrelevant |
| Prophet | additive time series | trend + annual seasonality + regressors |
| SARIMAX | ARIMA + exog | autocorrelated errors + Fourier seasonality |
| Seasonal median | calendar only | this week behaves like the same week in past years |
| Shape × level | calendar only | seasonal *shape* × typical annual *total* |

\normalsize

## Results --- holdout MAE

| Model | San Juan | Iquitos |
|---|---|---|
| **CatBoost** | **15.02** | 6.27 |
| RandomForest | 15.16 | 8.39 |
| XGBoost + RFE | 17.21 | 3.54 |
| **Shape × level** | 20.72 | **3.39** |
| Lasso | 21.16 | 4.50 |
| Prophet | 21.78 | 7.16 |
| Seasonal median | 23.49 | 4.41 |
| Ridge | 24.46 | 5.48 |
| SARIMAX | 24.56 | 4.61 |

Different winners per city --- the submission uses **CatBoost for San Juan, SARIMAX for
Iquitos**.

::: notes
XGBoost comes from a separate notebook (`experiments/xgboost_feature_selection.ipynb`)
that compared feature-selection strategies on chronological CV (3 folds of 52 weeks).
CV MAE: all 127 features 17.8 (San Juan) / 7.2 (Iquitos); the top 100 by XGBoost
importance 17.0 / 7.1; recursive elimination with Ridge 14.5 with 50 features / 6.7 with
10 features. On the same 52-week holdout the RFE model scores 17.21 and 3.54: third in
San Juan, and the best weather-based model in Iquitos, where ten long-window features
(8--26-week temperature means, 26-week rainfall and humidity, the annual sine and cosine)
suffice. A linear model chose features better than the trees' own importances did.
:::

## What drives the CatBoost predictions?

![CatBoost feature importance by feature family](figures/catboost_feature_importance.png){width=80%}

- **Temperature and humidity: about 70%** of importance in both cities
- Top features are **8--26-week averages**, not this week's weather
- **`year`: about 19%** --- levels drift between years for non-weather reasons

::: notes
Mosquitoes breed and bite more when it is warm and humid. Weather acts with a delay: the
mosquito population has to build up first, which is why the long windows win.

Rainfall matters less than expected (about 8%): temperature and humidity already capture
most of what the rain does.

Importance tells us what the model *uses*, not what *causes* dengue.
:::

## Why weather is not the whole story

- Climate predicts **when** cases rise within a year, not **how many**
- In Iquitos the calendar-only `Shape × level` (3.39) beats every weather model
- Epidemic size depends on the circulating **serotype** and population immunity ---
  neither is in the data

::: notes
Feature importance and the results table point the same way. A model that predicts the
shape of the year from the calendar and refuses to guess the level from weather is not
naive; it is honest about what the data supports.
:::

## Summary

- **Metrics** --- MAE, because the target is skewed (4.5): a few outbreak weeks would
  dominate RMSE
- **Models** --- nine models in four pipeline branches; winners differ by city
- **Feature importance** --- months of temperature and humidity; weather says *when*,
  not *how many*

::: notes
A side experiment with XGBoost confirms that feature selection matters: 50 features beat
127 in San Juan, and 10 are enough in Iquitos.
:::
