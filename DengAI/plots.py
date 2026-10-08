"""Presentation figures from the local data; no model training required.

Ported from the ``plot_utils.py`` module that ``presentation_part1`` used. Every function
draws one figure and calls ``plt.show()``; ``ndvi_missing_summary`` returns a table.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.model_selection import TimeSeriesSplit

from .config import load_config
from .data import load_raw

CITIES = {"sj": "San Juan", "iq": "Iquitos"}
COLORS = {"sj": "#236A91", "iq": "#D87832"}
NDVI = ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]


def load_data():
    """Training features merged with labels, sorted by city and week."""
    features, labels, _ = load_raw()
    data = features.merge(labels, on=["city", "year", "weekofyear"], validate="one_to_one")
    data["week_start_date"] = pd.to_datetime(data["week_start_date"])
    return data.sort_values(["city", "week_start_date"]).reset_index(drop=True)


def _axes(n=2, height=3.2, sharey=False):
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    return plt.subplots(1, n, figsize=(11.5, height), sharey=sharey, squeeze=False)


def city_patterns(data):
    fig, axes = _axes(height=3.5)
    ax, seasonal = axes[0]
    for city, name in CITIES.items():
        g = data[data.city.eq(city)]
        counts = np.sort(g.total_cases)
        ax.step(counts, np.arange(1, len(g)+1)/len(g)*100,
                color=COLORS[city], label=f"{name}: median {g.total_cases.median():.0f}")
        weekly = g[g.weekofyear.le(52)].groupby("weekofyear").total_cases
        profile = weekly.median()
        lower, upper = weekly.quantile(.25), weekly.quantile(.75)
        seasonal.fill_between(profile.index, lower, upper, color=COLORS[city],
                              alpha=.18, linewidth=0, label=f"{name}: 25–75%")
        seasonal.plot(profile.index, profile, color=COLORS[city],
                      label=f"{name}: median", lw=2)
    ax.set(xlabel="Weekly cases (log-spaced scale)", ylabel="Weeks at or below this count (%)",
           title="Different case distributions", xscale="symlog", xlim=(0, 500))
    ax.set_xticks([0, 5, 19, 100, 461], ["0", "5", "19", "100", "461"])
    seasonal.set(xlabel="Week of year", ylabel="Weekly cases",
                 title="Different seasonal patterns", xlim=(1, 52))
    seasonal.set_ylim(bottom=0)
    for a in axes[0]:
        a.legend(fontsize=10); a.grid(alpha=.15)
    fig.tight_layout(); plt.show()


def lag_evidence(data):
    columns = ["reanalysis_air_temp_k", "reanalysis_specific_humidity_g_per_kg",
               "precipitation_amt_mm", "ndvi_mean"]
    labels = ["Temperature", "Specific humidity", "Precipitation", "Vegetation (mean NDVI)"]
    fig, axes = _axes(height=3.3, sharey=True)
    for ax, (city, name) in zip(axes[0], CITIES.items()):
        g = data[data.city.eq(city)].copy()
        g["ndvi_mean"] = g[NDVI].mean(axis=1)
        for col, label in zip(columns, labels):
            correlations = []
            for lag in range(51):
                pairs = pd.DataFrame({"x": g[col].shift(lag), "y": g.total_cases}).dropna()
                correlations.append(pairs.x.rank().corr(pairs.y.rank()))
            ax.plot(range(51), correlations, label=label, lw=2)
        ax.axhline(0, color="gray", lw=.7)
        ax.axvspan(0, 26, color="#236A91", alpha=.04)
        ax.set(title=name, xlabel="Earlier observations (≈ weeks)", ylim=(-.65, .65), xlim=(0, 50))
    axes[0, 0].set_ylabel("Rank correlation with current cases")
    axes[0, 1].legend(fontsize=9, loc="lower right")
    fig.tight_layout(); plt.show()


def validation_timeline(data):
    """The chronological folds and holdout, read from configs/models.yaml."""
    ev = load_config().models.evaluation
    holdout, cv = ev.holdout_weeks, TimeSeriesSplit(n_splits=ev.cv_splits, test_size=ev.cv_weeks)
    fig, axes = _axes(height=3.1)
    for ax, (city, name) in zip(axes[0], CITIES.items()):
        n = len(data[data.city.eq(city)])
        for row, (train, valid) in enumerate(cv.split(np.arange(n-holdout))):
            ax.broken_barh([(0, len(train))], (row-.33, .66), facecolors=COLORS[city])
            ax.broken_barh([(valid[0], len(valid))], (row-.33, .66), facecolors="#69B8AA")
        last = ev.cv_splits
        ax.broken_barh([(0, n-holdout)], (last-.33, .66), facecolors=COLORS[city])
        ax.broken_barh([(n-holdout, holdout)], (last-.33, .66), facecolors="#A27DB8")
        ax.set(title=f"{name} · {n} observations", xlabel="Observation in chronological order",
               yticks=range(last+1),
               yticklabels=[f"CV {i+1}" for i in range(last)] + ["Final check"], xlim=(0, n))
        ax.invert_yaxis()
    fig.legend(handles=[Patch(color="#236A91", label="Training (expands)"),
                        Patch(color="#69B8AA", label=f"Next {ev.cv_weeks}: validation"),
                        Patch(color="#A27DB8", label=f"Last {holdout}: holdout")],
               loc="lower center", ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, 1)); plt.show()


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11.5, 2.3))
    ax.set(xlim=(-.15, 12), ylim=(0, 2)); ax.axis("off")
    labels = ["One city\nat a time", "Fill missing\nmeasurements", "Add weather\nhistory + season",
              "Model-specific\ninputs / selection", "Predict weekly\ncase counts"]
    for i, label in enumerate(labels):
        x = i*2.42
        ax.add_patch(FancyBboxPatch((x, .65), 2.1, .85, boxstyle="round,pad=.06",
                                    facecolor="#EAF2F6", edgecolor="#236A91"))
        ax.text(x+1.05, 1.075, label, ha="center", va="center", fontsize=12)
        if i < 4:
            ax.annotate("", xy=(x+2.36, 1.075), xytext=(x+2.12, 1.075),
                        arrowprops={"arrowstyle": "->", "color": "#236A91"})
    ax.text(6, .25, "Repeat the full recipe inside every validation fold",
            ha="center", fontsize=13, color="#236A91", weight="bold")
    fig.tight_layout(); plt.show()


def ndvi_missing_summary(data):
    """Missing percentages and longest consecutive row gaps, separately by city."""
    rows = []
    for city, name in CITIES.items():
        city_data = data[data.city.eq(city)].sort_values("week_start_date")
        for feature in NDVI:
            missing = city_data[feature].isna()
            runs = missing.ne(missing.shift()).cumsum()
            rows.append({"City": name, "Vegetation feature": feature,
                         "Missing (%)": round(100 * missing.mean(), 1),
                         "Longest gap (observations)": int(missing.groupby(runs).sum().max())})
    return pd.DataFrame(rows).set_index(["City", "Vegetation feature"])


def missingness_by_city(data):
    """Missing-feature percentages, matching presentation_eda's missingness chart."""
    missing = pd.DataFrame({name: data[data.city.eq(city)].isna().mean() * 100
                            for city, name in CITIES.items()})
    order = missing.max(axis=1).sort_values(ascending=False)
    missing = missing.loc[order[order.gt(0)].index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, (city, name) in zip(axes, CITIES.items()):
        bars = ax.barh(missing.index, missing[name], color=COLORS[city], height=.6)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in missing[name]], padding=5, fontsize=9)
        ax.set_title(name, loc="left", color=COLORS[city], weight="bold")
        ax.set_xlabel("Missing observations (%)")
        ax.set_xlim(0, missing.max().max() * 1.2)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Missing environmental measurements", fontsize=18, weight="bold")
    fig.tight_layout(); plt.show()
