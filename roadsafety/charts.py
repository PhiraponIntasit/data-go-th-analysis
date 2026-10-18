"""Static PNG charts for the report (matplotlib, English labels so they render anywhere).

Colour follows the period, the same in every chart:
Normal days = neutral grey, New Year = blue, Songkran = orange,
both festivals combined = aqua. Magnitude (heatmap) uses a single blue ramp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
COLORS = {
    "Normal days": "#9a9993",
    "New Year": "#2a78d6",
    "Songkran": "#eb6834",
    "Festival": "#1baf7a",
}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "figure.dpi": 110,
    }
)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _hbar(series: pd.Series, title: str, xlabel: str, color: str, fmt: str, out_dir: Path, name: str,
          ref: float | None = None, ref_label: str = "") -> Path:
    s = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.3))
    bars = ax.barh(s.index, s.values, color=color, height=0.65)
    for b, v in zip(bars, s.values):
        ax.annotate(fmt.format(v), (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    if ref is not None:
        ax.axvline(ref, color=INK_2, lw=1, ls="--")
        ax.annotate(ref_label, (ref, len(s) - 0.4), xytext=(4, 0), textcoords="offset points", color=INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.margins(x=0.12)
    return _save(fig, out_dir, name)


def recording_step(around: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(-0.5, 6.5, color=GRID, alpha=0.6, lw=0)
    ax.annotate("nominal 7-day campaign", (3, 0), xytext=(0, 6), textcoords="offset points", ha="center",
                color=INK_2, fontsize=9)
    for fest, g in around.groupby("period"):
        ax.plot(g["offset"], g["accidents"], lw=2, color=COLORS[fest], label=fest)
        peak = g.loc[g["accidents"].idxmax()]
        ax.annotate(fest, (peak["offset"], peak["accidents"]), xytext=(6, 4), textcoords="offset points",
                    color=INK, fontsize=9)
    ax.set_xlabel("Days from nominal campaign start (29 Dec / 11 Apr)")
    ax.set_ylabel("Recorded accidents per day (avg. across years)")
    ax.set_title("Recorded accidents jump 4× the day a campaign starts, and fall back the day it ends")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "01_recording_step.png")


def severity_by_period(rates: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(rates.index, rates["deaths_per_100_accidents"], color=[COLORS[p] for p in rates.index], width=0.6)
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                    va="bottom", xytext=(0, 3), textcoords="offset points", color=INK, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Deaths per 100 recorded accidents")
    ax.set_title("A festival crash is no deadlier than any other crash")
    ax.margins(y=0.15)
    return _save(fig, out_dir, "02_severity_by_period.png")


def festival_calendar(profile: pd.DataFrame, normal_deaths_per_day: float, out_dir: Path) -> Path:
    fests = [f for f in ["New Year", "Songkran"] if f in set(profile["period"])]
    fig, axes = plt.subplots(1, len(fests), figsize=(11, 3.8), sharey=True, squeeze=False)
    labels = {"12": "Dec", "01": "Jan", "04": "Apr"}
    for ax, fest in zip(axes[0], fests):
        g = profile[profile["period"] == fest]
        x = [f"{int(md[3:])} {labels[md[:2]]}" for md in g["md"]]
        bars = ax.bar(x, g["deaths_per_day"], color=COLORS[fest], width=0.65)
        top = g["deaths_per_day"].idxmax()
        for b, idx in zip(bars, g.index):
            if idx == top:
                ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                            va="bottom", xytext=(0, 3), textcoords="offset points", color=INK, fontsize=9)
        ax.axhline(normal_deaths_per_day, color=INK_2, ls="--", lw=1)
        ax.grid(axis="x", visible=False)
        ax.set_title(fest)
    axes[0][0].set_ylabel("Deaths per day (avg. across years)")
    axes[0][-1].legend(handles=[axes[0][-1].lines[0]], labels=["normal-day average"], loc="upper right")
    fig.suptitle("Which day of the campaign is deadliest?", x=0.01, ha="left", fontweight="bold", color=INK)
    return _save(fig, out_dir, "03_campaign_days.png")


def yearly(by_year: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    lasts = by_year.sort_values("festival_year").groupby("period").tail(1).set_index("period")["deaths_per_day"]
    for fest, g in by_year.groupby("period"):
        ax.plot(g["festival_year"], g["deaths_per_day"], marker="o", lw=2, ms=7, color=COLORS[fest], label=fest)
        last = g.iloc[-1]
        dy = 0 if len(lasts) < 2 else (7 if last["deaths_per_day"] >= lasts.drop(fest).max() else -7)
        ax.annotate(f"{fest} {last['deaths_per_day']:.1f}", (last["festival_year"], last["deaths_per_day"]),
                    xytext=(8, dy), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xticks(sorted(by_year["festival_year"].unique()))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Deaths per campaign day")
    ax.set_title("Deaths per campaign day, by year")
    ax.margins(x=0.15)
    ax.legend(loc="lower left")
    return _save(fig, out_dir, "04_deaths_by_year.png")


def hour_profile(hours: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for group, g in hours.groupby("group"):
        axes[0].plot(g["hour"], 100 * g["share_of_accidents"], lw=2, color=COLORS[group], label=group)
        axes[1].plot(g["hour"], g["deaths_per_100_accidents"], lw=2, color=COLORS[group], label=group)
    axes[0].set_title("When do accidents happen?")
    axes[0].set_ylabel("% of the period's accidents")
    axes[1].set_title("When are accidents deadliest?")
    axes[1].set_ylabel("Deaths per 100 accidents")
    for ax in axes:
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day")
        ax.set_ylim(bottom=0)
        ax.legend()
    return _save(fig, out_dir, "05_hour_profile.png")


def top_provinces(prov: pd.DataFrame, out_dir: Path, n: int = 15) -> Path:
    top = prov.nlargest(n, "festival_deaths").set_index("province_en")["festival_deaths"]
    return _hbar(top, f"Top {n} provinces by deaths during campaigns", "Deaths during New Year + Songkran campaigns",
                 COLORS["Festival"], "{:,.0f}", out_dir, "06_top_provinces.png")


def province_severity(prov: pd.DataFrame, overall: float, out_dir: Path, n: int = 15) -> Path:
    top = prov.dropna(subset=["festival_deaths_per_100_accidents"]).nlargest(n, "festival_deaths_per_100_accidents")
    s = top.set_index("province_en")["festival_deaths_per_100_accidents"]
    return _hbar(s, f"Top {n} provinces where a festival crash is most likely to kill",
                 "Deaths per 100 accidents during campaigns (provinces with 100+ accidents)",
                 COLORS["Festival"], "{:.1f}", out_dir, "07_province_severity.png",
                 ref=overall, ref_label="all provinces")


def vehicle_severity(vehicles: pd.DataFrame, out_dir: Path) -> Path:
    s = vehicles.drop(index=["Unknown", "Other"], errors="ignore")["deaths_per_100_accidents"].sort_values(ascending=False)
    return _hbar(s, "Deaths per 100 accidents by first vehicle involved", "Deaths per 100 accidents (all days)",
                 COLORS["Festival"], "{:.1f}", out_dir, "08_vehicle_severity.png")


def causes(table: pd.DataFrame, out_dir: Path) -> Path:
    s = 100 * table.drop(index=["Unknown", "Other"], errors="ignore")["share_Festival"].sort_values(ascending=False)
    return _hbar(s, "Presumed cause of accidents during campaigns", "% of campaign accidents",
                 COLORS["Festival"], "{:.1f}%", out_dir, "09_causes.png")


def fatal_map(df: pd.DataFrame, out_dir: Path) -> Path:
    pts = df.dropna(subset=["latitude", "longitude"])
    fest_fatal = pts[(pts["period"] != "Normal days") & pts["is_fatal"]]
    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.scatter(pts["longitude"], pts["latitude"], s=1, color=GRID, alpha=0.6, label="All recorded accidents", rasterized=True)
    ax.scatter(fest_fatal["longitude"], fest_fatal["latitude"], s=9, color=COLORS["Festival"], alpha=0.7,
               edgecolor="none", label="Fatal accidents during campaigns")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.set_title("Where festival deaths happen")
    ax.legend(loc="lower left", markerscale=1.5)
    return _save(fig, out_dir, "10_fatal_map.png")
