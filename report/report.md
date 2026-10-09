# DengAI: Predicting Weekly Dengue Cases from Environmental Data

**Team:** Artem Perepelitsyn, Nurdaulet Akhanov, Aibota Sanatbyek

## 1. The problem and why it matters

### 1.1 Dengue and the forecasting task

Dengue is a mosquito-borne viral disease of tropical and subtropical regions; severe cases
can be fatal. Outbreaks follow the weather: warmth, humidity and rainfall drive mosquito
breeding, so case counts rise and fall with the climate, often with a delay of weeks.
Anticipating those surges is hard, and the consequences of getting it wrong are
concrete.

This project addresses the DrivenData *DengAI* competition: **predict the number of
dengue cases reported each week** in San Juan (Puerto Rico) and Iquitos (Peru) from
environmental measurements — temperature, humidity, precipitation and satellite
vegetation indices. It is a supervised regression problem on two weekly time series.

### 1.2 How a forecasting model helps

A working forecast helps in three ways:

- **Earlier warning.** A surge can be predicted before it shows up in clinics.
- **Lower cost.** Spraying and breeding-site clean-up can be targeted to the weeks and
  places where they are needed, instead of being spread evenly.
- **Better care.** Hospitals can prepare beds, staff and test kits ahead of a peak.

### 1.3 Why a score is not a decision

The last point is the one we keep returning to in the analysis (Section 6). If a model's
forecast is used by, say, the Regional Hospital of Loreto in Iquitos to plan medicine
purchases, an **underprediction** means too few medications and direct risk to patients,
while an **overprediction** means expired supplies, wasted money and capacity taken from
other departments. The competition scores a number; a hospital makes a decision. Keeping
that distinction in view shaped how we evaluated the models and how far we are willing to
trust them.

## 2. Loading the data and splitting it into inputs and outputs

### 2.1 The three files

The competition provides three CSV files, loaded with `pandas` through the project's
`load_raw()` helper: training features, training labels and test features. They are
joined on the key `(city, year, weekofyear)`, which identifies one week in one city.

| Table | Shape | Content |
|---|---|---|
| training features | 1,456 × 24 | 20 environmental measurements plus `city`, `year`, `weekofyear`, `week_start_date` |
| training labels | 1,456 × 4 | the key plus `total_cases`, the weekly count of reported cases |
| test features | 416 × 24 | the same columns for the future weeks to predict |

### 2.2 Inputs, output and the per-city split

The **output `y` is `total_cases`**; the **inputs `X` are the 20 environmental columns**
together with the date information needed to build features from them. The two cities
differ in history and scale, so after loading the frames are split by city and every
subsequent step — imputation, feature engineering, validation, model fitting — is done
**per city**:

| City | Training weeks | Training period | Test weeks | Test period |
|---|---|---|---|---|
| San Juan (`sj`) | 936 | 1990-04-30 → 2008-04-22 | 260 | 2008-04-29 → 2013-04-23 |
| Iquitos (`iq`) | 520 | 2000-07-01 → 2010-06-25 | 156 | 2010-07-02 → 2013-06-25 |

### 2.3 Two timelines and a pure future hold-out

Each city's test period follows directly after its training period — about five years
for San Juan and three for Iquitos. The test set is therefore a *pure future hold-out*,
which is the single most important fact about this dataset: whatever we do must be
validated on later weeks, never on a random sample.

### 2.4 Duplicate check

Before anything else the notebook runs a **duplicate check** on all three tables: no
duplicated rows, no duplicated `(city, year, weekofyear)` keys or dates, exactly one label
per feature row, and no two columns carrying the same values. The only hit is a column:
`reanalysis_sat_precip_amt_mm` is a byte-for-byte copy of `precipitation_amt_mm` in both
train and test, so it is dropped.

## 3. Preparing the dataset

### 3.1 Categorical variables

The only categorical column is `city`, and it never reaches a model: the two cities are
modelled separately, so there is nothing to encode. `week_start_date` is used to derive
features and is then dropped, together with `weekofyear` (replaced by the cyclical
encoding below) and the duplicate precipitation column.

### 3.2 Missing values

Vegetation has by far the most gaps, and the pattern differs between cities:

| City | Feature | Missing (%) | Longest gap (consecutive weeks) |
|---|---|---|---|
| San Juan | `ndvi_ne` | 20.4 | 15 |
| San Juan | `ndvi_nw` | 5.2 | 15 |
| San Juan | `ndvi_se` / `ndvi_sw` | 2.0 | 14 |
| Iquitos | all four `ndvi_*` | 0.6 | 1 |

The `ndvi_ne` pixel of San Juan falls over the Atlantic, which is why it is missing a
fifth of the time. This motivates two different filling rules, implemented in one
transformer (`DengueImputer`) that is *fitted on training weeks only*:

| Columns | San Juan | Iquitos | Why |
|---|---|---|---|
| Vegetation (`ndvi_*`) | **Median of the same week of the year**, learned from training data; overall training median as fallback | Linear interpolation | San Juan's gaps run up to 15 weeks. Forward-filling would repeat one stale value for months; the same-week median keeps the seasonal shape. Iquitos' gaps are a single week, where interpolation is exact enough |
| All other weather columns | Linear interpolation between the nearest known values | Linear interpolation | Gaps are short; temperature and humidity change smoothly from week to week |

The test features are filled with the *already fitted* imputer (`transform` only), so no
information from the test period enters the filling rules.

### 3.3 New features

Two findings from the exploratory analysis (Section 4) drive the feature engineering.

**Earlier weather explains current cases.** The effect of weather on cases is delayed:
in San Juan, temperature and humidity correlate most with cases **8–10 weeks earlier**,
because the mosquito population has to build up first. The `WeatherHistoryTransformer`
therefore adds, for the main temperature, humidity and rainfall columns:

- **lagged values** from 1, 2, 4, 8, 10, 20 and 26 weeks earlier;
- **rolling summaries** over the previous 4, 8, 10, 12, 16 and 26 weeks — *means* for
  temperature and humidity, *sums* for rainfall, because rain accumulates rather than
  averages.

Every window is shifted by one week before rolling, so a week's own measurement is never
inside its own history, and the transformer carries the last 26 training weeks forward so
that the first test weeks have a complete history. Altogether the 20 raw measurements
become about 125 model inputs.

**The year is a circle.** Week 52 and week 1 are neighbours, but as integers they are 51
apart. Each week is placed on a circle with `sin(2π·t)` and `cos(2π·t)` of the fraction of
the year elapsed, so that late December and early January look alike to the models.

### 3.4 The target

`total_cases` is left as a raw count. The evaluation metric is the mean absolute error on
case counts, so transforming the target (for example to a logarithm) would optimise the
wrong quantity; and the count-like, right-skewed shape of the target is exactly what the
metric choice in Section 5 is built around.

### 3.5 Everything is a pipeline

All of the above — filling rules, lags, scaling, feature selection — is fitted inside a
scikit-learn `Pipeline`, one per city and model, and refitted inside every validation
fold. This is what makes the validation honest: validation case counts never enter
training, and the same fitted preparation is reused unchanged on new data.

## 4. Exploratory analysis

The exploratory analysis was done per city, with Seaborn heatmaps of the full
correlation matrix, bar charts of each feature's correlation with the target, and the
distribution, seasonality, lag and missingness figures drawn by the notebook's
`DengAI.plots` module.

### 4.1 Two cities, two problems

San Juan averages 34.2 cases a week with a
median of 19 and a maximum of 461; Iquitos averages 7.6 with a median of 5 and a maximum
of 116. Their seasons differ too: San Juan's cases climb through the second half of the
year and peak around weeks 35–45, while Iquitos' outbreaks fall between October and April.
Decision: perform EDA, preparation and modelling separately for each city.

### 4.2 A heavily skewed target

Most weeks are quiet and a few outbreak weeks carry
hundreds of cases. Skewness is **4.48 in San Juan and 4.00 in Iquitos** (zero would be a
symmetric bell curve; above one already counts as strongly skewed), 73% of San Juan's
weeks lie below its mean, and in both cities the worst 10% of weeks hold **43% of all
cases**. The variance is far larger than the mean (San Juan 2,640 vs 34.2; Iquitos 115.9
vs 7.6) — the overdispersion that motivated the competition's own negative-binomial
benchmark. This finding decides the metric (Section 5.1).

### 4.3 No single feature predicts cases, and the signal is delayed

No environmental
column correlates strongly with `total_cases`. The rank correlation between a weather
variable and the current week's cases is, however, systematically higher when the
variable is taken from several weeks earlier, peaking at 8–10 weeks for temperature and
humidity in San Juan. This is the evidence behind the lag and rolling-window features.

### 4.4 Strong week-to-week autocorrelation

Weekly cases are strongly autocorrelated (0.97 from one week to the next). A random
train/test split would put week *t* in training and week *t+1* in validation and let the
model memorise the answer; validation must be chronological (Section 5.2).

### 4.5 Redundant inputs

The reanalysis temperatures correlate strongly with
each other and with the station temperatures; specific humidity tracks dew-point
temperature. Strong correlation suggests overlap but not identity, so the only column
removed on these grounds is the one that is an exact copy (`reanalysis_sat_precip_amt_mm`);
the rest is left to feature selection inside the models.

### 4.6 Climate explains when, not how much

The seasonal profile is reproducible from
year to year, but the height of the epidemic years is not visible in the weather. This
observation led us to add two calendar-only baselines (Section 5.3) that any
weather-driven model has to beat, and it turns out to be the central result of the
project.

## 5. Validation, metric, models and tuning

### 5.1 Why mean absolute error

The competition scores submissions by mean absolute error (MAE): the average number of
cases per week the prediction is off by, pooled over all 416 test weeks. We adopted it for
validation as well, and not only because it is the competition's metric. With a target
this skewed, RMSE would be dominated by a handful of outbreaks — a single week with 461
cases contributes as much to RMSE as about 500 ordinary weeks that are each off by one.
MAE weights every week equally; statistically it is minimised by the conditional median
while RMSE is minimised by the mean, and for a skewed count the median is the more robust
target. One consequence, visible later, is that models which do well on this metric are
*conservative*: they predict the normal level and rarely chase peaks.

Because the two cities are pooled into one number, San Juan dominates it: 63% of the test
weeks and roughly four times the case counts.

### 5.2 Chronological cross-validation and holdout

Each city's series is split in time order:

- the **last 52 weeks** are set aside as a holdout block that no model sees during tuning;
- on the remaining weeks, `TimeSeriesSplit(n_splits=5, test_size=52)` gives five folds,
  each training on all earlier weeks and validating on the next 52 (an expanding window).

Hyperparameters are chosen by the **CV MAE** (the tuning signal); models are then compared
by the **holdout MAE** (the honest estimate). The two are never equal, and a large gap
between them is itself information about overfitting. The constraint is enforced in code,
not just intended: the SARIMAX wrapper raises an exception if asked to predict a block that
does not start strictly after its training data. Nothing in the project shuffles.

### 5.3 The models

Different model families need different preprocessing, so the pipeline builder has five
branches, and nine models were compared on identical folds and holdouts:

| Branch | Steps after imputation and weather history | Models |
|---|---|---|
| `tree` | season encoding → CatBoost SHAP feature selection → model | CatBoost, RandomForest |
| `dense` | the tree steps, then median imputation of the lag warm-up + standard scaling | RidgeCV, LassoCV |
| `rfe` | imputation + scaling → recursive feature elimination driven by Ridge → model | XGBoost |
| `time_series` | date + four scaled weather regressors (4-week means of temperature, specific and relative humidity, 4-week rainfall sum) | Prophet, SARIMAX |
| `seasonal` | calendar columns only (`year`, `weekofyear`) | Seasonal median, Shape × level |

`dense` exists because the lag features leave NaNs in the first 26 weeks of history,
which tree models tolerate natively but `RidgeCV` rejects; scaling matters for the linear
models in any case because their penalties are not scale-invariant. `rfe` exists because
the selector that won the feature-selection experiment cannot see NaN either, so there
selection has to come *after* imputing — the opposite order from the CatBoost selector.
`seasonal` exists because the calendar baselines need exactly the columns the tree branch
drops, and because they never see the weather: they are the baseline any weather model
must beat.

| Model | Family | Core assumption |
|---|---|---|
| CatBoost | gradient boosting | non-linear effects of lagged weather |
| XGBoost | gradient boosting | as CatBoost, on Ridge-RFE-selected features |
| RandomForest | bagged trees | same, without boosting |
| Ridge | linear + L2 | smooth linear response, all features shrunk |
| Lasso | linear + L1 | linear, but most features are irrelevant |
| Prophet | additive time series | trend + annual seasonality + weather regressors |
| SARIMAX | ARIMA + exogenous inputs | autocorrelated errors + Fourier seasonality |
| Seasonal median | calendar only | this week behaves like the same week in past years |
| Shape × level | calendar only | seasonal *shape* (normalised to sum 1) × typical annual *total* |

### 5.4 Hyperparameter search

Every model's grid is declared in `configs/models.yaml` and searched with
`GridSearchCV` over the five chronological folds, scored by negative MAE:

| Model | Grid |
|---|---|
| CatBoost | iterations {500, 1000} × depth {3, 4, 6} × learning rate 0.02, with the SHAP selector keeping {5, 7, 10, 15, 20, 30, 40, 50, 60} features or switched off (60 candidates) |
| XGBoost | depth {2, 4, 6} × learning rate {0.03, 0.1} × trees {100, 300} × L2 {1, 10} × RFE keeping {10, 25, 50, 100} features (96 candidates) |
| RandomForest | max features {0.33, √p} × min samples per leaf {1, 3} |
| Ridge / Lasso | penalty strength chosen internally (`RidgeCV` over 0.01–10,000; `LassoCV` on a 5-fold path) |
| Prophet | yearly seasonality order {3, 5} × changepoint prior {0.01, 0.05} |
| SARIMAX | order {(1,0,0), (2,0,0), (1,0,1)} × Fourier order {2, 3} |
| Seasonal median | statistic {median, mean} × smoothing window {1, 5, 9} |
| Shape × level | shape statistic {mean, median} × smoothing window {1, 5, 9} |

Selected hyperparameters: CatBoost San Juan — depth 6, 1,000 iterations, learning rate
0.02, 15 selected features; CatBoost Iquitos — depth 3, 500 iterations, learning rate 0.02,
40 features; XGBoost San Juan — depth 6, 300 trees, learning rate 0.03, L2 = 1, 25 RFE
features; XGBoost Iquitos — depth 4, 100 trees, learning rate 0.03, L2 = 10, 25 features.
Prophet keeps yearly seasonality 5 in both cities; SARIMAX chose AR(1) with Fourier order 2
in San Juan and AR(2) with Fourier order 3 in Iquitos.

### 5.5 Results

Holdout MAE on each city's final 52 weeks, lower is better (CV MAE in brackets):

| Model | San Juan | Iquitos |
|---|---|---|
| XGBoost + Ridge-RFE | **13.94** (14.20) | 4.17 (7.85) |
| CatBoost | 15.02 (12.23) | 6.27 (7.40) |
| RandomForest | 15.16 (19.97) | 8.39 (7.53) |
| Shape × level | 20.72 (16.90) | **3.39** (7.32) |
| Lasso | 21.42 (23.52) | 4.44 (7.94) |
| Prophet | 21.78 (17.18) | 7.16 (8.14) |
| Seasonal median | 23.49 (15.51) | 4.41 (6.99) |
| Ridge | 24.46 (33.29) | 5.48 (9.30) |
| SARIMAX | 24.56 (21.43) | 4.61 (6.98) |

For reference, a constant prediction of the training median scores 27.50 in San Juan and
3.85 in Iquitos on the same holdouts.

#### What the table says

Three things stand out, independently of the exact decimals:

1. **The winners differ by city.** In San Juan the gradient-boosting models on lagged
   weather are clearly best, and feature selection matters: XGBoost with Ridge-RFE keeping
   25 features edges out CatBoost. In Iquitos the best model is **Shape × level — a
   calendar-only baseline that never looks at the weather**; every weather-driven model is
   behind it.
2. **The submission uses CatBoost for San Juan and SARIMAX for Iquitos**, the choice made
   when the three original models were compared, before the six additional models were
   added for comparison. The additional experiments do not change the submission; they
   change how we read it (Section 6).
3. **What CatBoost uses.** Grouping its feature importances by what each feature
   measures: temperature and humidity carry about 70% of the importance in both cities;
   every top feature is a long average over 8–26 weeks rather than this week's weather;
   rainfall contributes only about 8%; and the `year` column alone takes about 19%, i.e.
   case levels drift from year to year for reasons that are not weather.

### 5.6 A note on the public leaderboard

Four of the additional models were at some point submitted to the real DrivenData
leaderboard from earlier, standalone feature sets (RandomForest 23.80, Ridge 27.00,
seasonal median 26.46, shape × level 26.00); the final CatBoost/SARIMAX submission scores
about 23. These numbers are not reproducible from the notebook's single
consistent pipeline, and they disagree with the holdout ranking in an instructive way.
Across the submitted models, local validation MAE correlated only ρ = −0.30 with the
leaderboard score — mildly *inverted* — whereas the number of weeks a model was willing to
predict above 50 cases correlated ρ = −0.90. Every validation window is drawn from
ordinary years, so it rewards conservative prediction; the competition's test period is
dominated by Puerto Rico's 2010 epidemic, the largest since surveillance began, which
punishes exactly that. A model can be better on our holdout and worse on the
leaderboard, and this is a property of the data.

## 6. Analysis of the results, real-world applicability and conclusion

### 6.1 What the experiments show

The experiments confirm that weekly dengue incidence is a difficult forecasting problem.
In both cities the variance of the target is much larger than its mean (San Juan 2,640 vs
34.2; Iquitos 115.9 vs 7.6). This overdispersion motivated the competition's original
negative-binomial solution and, combined with the visibly ordered epidemic waves in the
data, gave us a reason to test whether models that explicitly use temporal dependence —
SARIMAX, Prophet, and the calendar baselines — could compete with models that learn from
the weather.

The most important finding is that **the best model is different for each city, and in
Iquitos the best model does not use the weather at all.** Among the three original
models, SARIMAX gave the lowest Iquitos holdout MAE (4.61 against CatBoost's 6.27 and
Prophet's 7.16), which is why it is the Iquitos model in the submission. The additional
experiments then showed that a seasonal shape scaled by a typical annual total — a model
with no inputs beyond the calendar — scores 3.39, better than any weather-based model
in Iquitos. For San Juan the boosted tree models on lagged weather remain clearly best
(XGBoost 13.94, CatBoost 15.02, against 20.72 for the calendar baseline), so there the
weather does carry usable signal. A single universal model is not appropriate for both
cities.

The experiments also reveal a weakness of MAE as the only yardstick. It summarises the
average error but does not show *when* errors occur or distinguish overprediction from
underprediction. Although CatBoost achieved a low San Juan holdout MAE, the holdout plot
shows it substantially underestimating the largest peak of that year. A model can obtain a
reasonable average score while failing in exactly the weeks when an accurate forecast
matters most. The leaderboard note above is the same phenomenon at a larger scale.

Prophet performed worse than both CatBoost and SARIMAX in both cities. Its assumed trend
plus seasonal structure appears too restrictive for abrupt, irregular epidemic dynamics.
This does not show that Prophet is unsuitable for dengue forecasting in general — its
performance also depends on the history window, the seasonal terms and how exceptional
outbreaks are treated — but it does show that a standard seasonal forecast should not be
accepted without city-specific tuning and evaluation on epidemic periods.

### 6.2 What the model captures and what it misses

The Iquitos result creates the central tension of the project. The models that do best
there gain their accuracy from autocorrelation and recurring seasonal structure, not from
the weather variables; and the feature-importance analysis says the same about CatBoost
in a quieter way — its strongest inputs are months-long averages that track the season,
and a fifth of its importance sits on the `year` column, which encodes nothing about
climate. **Forecasting improvement and evidence about the underlying cause are separate
questions.** Climate reliably tells the models *when* within a year cases rise; it tells
them very little about *how many*.

Every predictive feature in the DengAI dataset describes climate or vegetation:
temperature, humidity, precipitation, NDVI. Dengue, however, is a chain — weather →
mosquito abundance → infection → reported cases — and the model observes only the first
and the last link. It has no information about mosquito abundance or infection rates,
population immunity, circulating dengue serotypes, vector-control interventions, human
movement, reporting practices, or damage caused by natural disasters. DrivenData's own
launch post calls the relationship between dengue and climate complex.

This reading is supported by the study *Long-Term and Seasonal Dynamics of Dengue in
Iquitos, Peru* (ten years of laboratory-confirmed surveillance, 2000–2010). It found
transmission dominated by single serotypes — DENV-3 in 2001–2007, then DENV-4 in
2008–2010 — seasonal peaks between October and April despite limited intra-annual
variation in climate, a strong positive autocorrelation in case counts at a lag of about
70 weeks, and only weak correlations between every climatic variable and reported cases
across a range of lags; early city-wide insecticide fumigation, on the other hand,
appeared to reduce transmission in some years. Thus the city where it was easiest for us
to score well is also the city where the published evidence most clearly warns against
interpreting a climate-based forecast as an explanation of dengue dynamics.

Adding mosquito surveillance data would be the natural next step, but the required
historical coverage is not readily available. The Iquitos mosquito-abundance series
(Reiner et al., 2019) covers 1999–2010, while the DengAI period starts earlier; for San
Juan, the identified mosquito datasets were collected from 2017 onwards (Barrera et al.,
2019; Yee & Scavo, 2021; Otero et al., 2025), long after the 1990–2008 training period,
and some are available only on request. Adding mosquito variables to the historical model
is therefore not a simple feature-engineering step, because the relevant weekly
observations do not align with the target period. The very existence of these continuing
surveillance programmes shows that practitioners understand the need for more directly
related data than climate.

Natural disasters are another source of unobserved variation. Barrera et al. (2019)
describe how severe storms alter mosquito habitats by filling containers, changing
groundwater edges and creating breeding sites in damaged or abandoned properties, so
disaster dates, rainfall, flooding, housing damage and displacement could improve the
description of exceptional periods. The largest San Juan peak in the data, in 1994,
occurred in the year of the *Morris J. Berman* oil spill (7 January 1994). We found no
study establishing a causal link between the spill and that outbreak, however, this link seems plausible.

### 6.3 Applicability in a real-life scenario

The model could serve as a decision-support tool for short-term planning. The Regional
Hospital of Loreto, for instance, could use a weekly forecast as one input when planning
medicine purchases, staffing, bed capacity and public-health communication. The Iquitos
holdout MAE of 4.61 for the submitted model means its prediction differed from the
observed count by about 4.6 cases per week on average; for San Juan the submitted
model differed by about 15 cases per week.

These values are averages, not guarantees. Underprediction could leave a hospital with
insufficient medicine, staff or beds, directly affecting patient care; overprediction
could cause unnecessary purchases, expired supplies and resources diverted from other
departments. The two errors have different consequences even though MAE weights them
equally, and the average hides the peak weeks where the model is weakest. The model
should therefore not determine inventory or capacity on its own. It is better suited to
producing an early warning that professionals combine with current surveillance data
and clinical judgement.

Before real deployment the system would need several additions: weekly mosquito trap
counts and infection rates for the same years as the cases; serotype data and serosurveys
describing immunity; intervention logs of when and where vector control happened;
structured natural-disaster data (storm dates and tracks, rainfall and flood extent,
housing damage); quantile forecasts or prediction intervals with verified coverage
instead of a point forecast tuned for MAE; and a test of our own claim — the same model
with and without climate covariates, and with a disaster log added, compared on the
epidemic years. A prospective pilot should measure not only forecasting error but also
medicine shortages, expired inventory, bed occupancy and the number of high-incidence
weeks detected early enough for action.

### 6.4 Overall conclusion and estimated impact

The model partially solves the problem stated at the beginning. It forecasts the weekly
dengue curve, the experiments show measurable differences between models on honest
chronological holdouts, and the city-specific results are practically useful: boosted
trees on lagged weather for San Juan, temporal structure — SARIMAX in the submission, and
the calendar itself in the strongest holdout result — for Iquitos. In this limited
predictive sense the project succeeds, and it does so with validation that cannot leak
the future into the past.

It does not yet solve the broader problem of making reliable hospital decisions during
dengue outbreaks. Its inputs are too narrow, its point forecasts carry no uncertainty,
its average error hides failures at epidemic peaks, and a validation built from ordinary
years can rank models differently from a test period dominated by an epidemic. It
predicts patterns without establishing why they occur: a model that has never seen a
mosquito cannot say what happens when the mosquitoes change. It should not be described
as a causal model or deployed as the sole basis for allocating medical resources.

The likely impact is therefore positive but conditional. Used as an additional warning
signal, the forecasts could give hospitals and public-health teams more time to prepare
staff, medicines and beds, reducing shortages during rising transmission and limiting
waste in quieter periods. A numerical estimate of lives saved or costs avoided cannot be
justified from the DengAI data alone: the dataset contains no procurement costs, hospital
capacities, treatment outcomes or records of decisions taken from forecasts. Estimating
that impact would require a prospective pilot or a retrospective simulation connecting
forecasts to explicit hospital actions and their costs. The current work is best
understood as a promising forecasting component that needs broader epidemiological data,
calibrated uncertainty and operational validation before it can become a dependable
public-health decision system.

## References

1. DrivenData, *DengAI: Predicting Disease Spread* — competition page, feature
   definitions and benchmark post (https://drivendata.co/blog/dengue-benchmark/).
2. Stoddard, S. T. et al. *Long-Term and Seasonal Dynamics of Dengue in Iquitos, Peru.*
   PLoS Neglected Tropical Diseases, DOI 10.1371/journal.pntd.0003003.
3. Reiner, R. C. et al. (2019). Iquitos mosquito-abundance data, 1999–2010.
4. Barrera, R. et al. (2019). Post-hurricane mosquito surveillance in metropolitan San
   Juan after Hurricanes Irma and Maria.
5. Yee, D. A. & Scavo, N. A. (2021). Mosquito-community data, San Juan, 2018–2019.
6. Otero, L. et al. (2025). Mosquito surveillance during the 2024 San Juan dengue epidemic.
7. Puerto Rico Vector Control Unit (n.d.). Bayamón mosquito-control and surveillance
   programme.
8. Data sources: NOAA GHCN daily weather stations; PERSIANN satellite precipitation;
   NOAA NCEP Climate Forecast System Reanalysis; NOAA CDR Normalized Difference
   Vegetation Index.
