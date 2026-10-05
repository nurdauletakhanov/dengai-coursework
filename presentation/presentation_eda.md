# Additional EDA

# EDA setup

Weekly case counts per city:

| City | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| Iquitos | 520 | 7.57 | 10.77 | 0 | 1 | 5 | 9 | 116 |
| San Juan | 936 | 34.18 | 51.38 | 0 | 9 | 19 | 37 | 461 |

## Weekly cases

![Weekly dengue cases over time, by city](presentation_eda_files/presentation_eda_7_0.png)

## Seasonality

![Seasonal profile: median cases by week of year](presentation_eda_files/presentation_eda_9_0.png)

San Juan tends to have higher counts later in the year, while Iquitos has higher medians near the beginning. This supports exploring seasonal features separately for the cities.

## Weather and cases

![Weather and vegetation against case counts](presentation_eda_files/presentation_eda_12_0.png)

## Missing values

![Missing values by column and city](presentation_eda_files/presentation_eda_14_0.png)

## Weather lags

Compare earlier weather and vegetation with current cases using Spearman correlation. A lag of 4 means four rows earlier, approximately four weeks.

![Spearman correlation of lagged weather with current cases](presentation_eda_files/presentation_eda_16_0.png)

