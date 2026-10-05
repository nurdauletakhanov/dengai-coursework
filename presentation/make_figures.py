"""Draw the two figures in presentation/figures/.

  target_skew.png                  histogram of weekly cases (training labels)
  catboost_feature_importance.png  importance of the fitted CatBoost pipelines

Run from the repo root with the project venv: .venv/bin/python presentation/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from DengAI.evaluate import load_pipelines
from DengAI.paths import ROOT

FIGURES = Path(__file__).parent / "figures"
OUT = FIGURES / "catboost_feature_importance.png"
BLUE = "#2a6fb0"
CITIES = {"sj": "San Juan", "iq": "Iquitos"}
ORDER = ["Temperature", "Humidity", "Year (trend)", "Rainfall", "Season (calendar)"]


def family(name):
    if name == "year":
        return "Year (trend)"
    if "humid" in name or "dew" in name:
        return "Humidity"
    if "temp" in name:
        return "Temperature"
    if "precip" in name:
        return "Rainfall"
    return "Season (calendar)"


pipelines = load_pipelines("catboost", warn=False)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharex=True)
for ax, (city, label) in zip(axes, CITIES.items()):
    model = pipelines[city].named_steps["model"]
    imp = pd.Series(model.get_feature_importance(), index=model.feature_names_)
    shares = imp.groupby(imp.index.map(family)).sum().reindex(ORDER).fillna(0)[::-1]
    ax.barh(shares.index, shares.values, color=BLUE, height=0.6)
    for y, v in enumerate(shares.values):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=9, color="#333333")
    ax.set_title(f"{label} ({len(imp)} features)", fontsize=11, color="#222222", loc="left")
    ax.set_xlim(0, 55)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="y", length=0, labelsize=10, colors="#333333")
fig.suptitle("Share of CatBoost feature importance, grouped by what the feature measures",
             fontsize=11, color="#222222", x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")


# Weekly cases: most weeks are quiet, a few outbreak weeks form a long right tail.
labels = pd.read_csv(ROOT / "data" / "dengue_labels_train.csv")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, (city, label) in zip(axes, CITIES.items()):
    cases = labels.loc[labels.city == city, "total_cases"]
    ax.hist(cases, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
    for value, name, style in ((cases.median(), "median", "-"), (cases.mean(), "mean", "--")):
        ax.axvline(value, color="#222222", linestyle=style, linewidth=1.2)
        ax.text(value, ax.get_ylim()[1] * (0.92 if name == "median" else 0.78),
                f" {name} {value:.0f}", fontsize=9, color="#222222")
    ax.set_title(f"{label} (skewness {cases.skew():.1f})", fontsize=11, color="#222222", loc="left")
    ax.set_xlabel("cases in a week", fontsize=9, color="#333333")
    ax.set_ylabel("number of weeks", fontsize=9, color="#333333")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors="#333333")
fig.tight_layout()
out = FIGURES / "target_skew.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")
