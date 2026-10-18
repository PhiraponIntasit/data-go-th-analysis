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


def _bar_labels(ax, bars, fmt: str = "{:,.0f}") -> None:
    for b in bars:
        ax.annotate(
            fmt.format(b.get_height()),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", color=INK, fontsize=9,
        )


def daily_rates(rates: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [
        (axes[0], "accidents_per_day", "Accidents per day"),
        (axes[1], "deaths_per_day", "Deaths per day"),
    ]:
        bars = ax.bar(rates.index, rates[col], color=[COLORS[p] for p in rates.index], width=0.6)
        _bar_labels(ax, bars, "{:,.1f}")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    fig.suptitle("Festival days vs normal days (Department of Highways network)", x=0.01, ha="left", color=INK_2, fontsize=10)
    return _save(fig, out_dir, "01_daily_rates.png")


def festival_days(profile: pd.DataFrame, normal_deaths_per_day: float, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for fest, g in profile.groupby("period"):
        ax.plot(g["festival_day"], g["deaths_per_day"], marker="o", lw=2, ms=6, color=COLORS[fest], label=fest)
        last = g.iloc[-1]
        ax.annotate(fest, (last["festival_day"], last["deaths_per_day"]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.axhline(normal_deaths_per_day, color=COLORS["Normal days"], ls="--", lw=1.5)
    ax.annotate("normal-day average", (1, normal_deaths_per_day), xytext=(0, 4), textcoords="offset points",
                color=INK_2, fontsize=9)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Day of the 7-day campaign window")
    ax.set_ylabel("Deaths per day (average across years)")
    ax.set_title("Which day of the festival is the deadliest?")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.7, 7.9)
    ax.legend(loc="lower left")
    return _save(fig, out_dir, "02_festival_days.png")


def hour_profile(hours: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for group, g in hours.groupby("group"):
        axes[0].plot(g["hour"], 100 * g["share_of_accidents"], lw=2, color=COLORS[group], label=group)
        axes[1].plot(g["hour"], g["deaths_per_100_accidents"], lw=2, color=COLORS[group], label=group)
    axes[0].set_title("When do accidents happen?")
    axes[0].set_ylabel("% of accidents in the period")
    axes[1].set_title("When are accidents deadliest?")
    axes[1].set_ylabel("Deaths per 100 accidents")
    for ax in axes:
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day")
        ax.set_ylim(bottom=0)
        ax.legend()
    return _save(fig, out_dir, "03_hour_profile.png")


def weekday_hour_heatmap(matrix: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(matrix.values, aspect="auto", cmap=BLUES)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.grid(False)
    ax.set_title("Accidents by weekday and hour (all days)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("Accidents", color=INK_2)
    return _save(fig, out_dir, "04_weekday_hour_heatmap.png")


def _hbar(series: pd.Series, title: str, xlabel: str, color: str, fmt: str, out_dir: Path, name: str,
          ref: float | None = None) -> Path:
    s = series.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.2))
    bars = ax.barh(s.index, s.values, color=color, height=0.65)
    for b, v in zip(bars, s.values):
        ax.annotate(fmt.format(v), (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    if ref is not None:
        ax.axvline(ref, color=INK_2, lw=1, ls="--")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.margins(x=0.12)
    return _save(fig, out_dir, name)


def top_provinces(prov: pd.DataFrame, out_dir: Path, n: int = 15) -> Path:
    top = prov.nlargest(n, "festival_deaths").set_index("province_en")["festival_deaths"]
    return _hbar(top, f"Top {n} provinces by festival deaths", "Deaths during New Year + Songkran windows",
                 COLORS["Festival"], "{:,.0f}", out_dir, "05_top_provinces.png")


def province_lift(prov: pd.DataFrame, out_dir: Path, n: int = 15) -> Path:
    top = prov.dropna(subset=["lift"]).nlargest(n, "lift").set_index("province_en")["lift"]
    if top.empty:
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.axis("off")
        ax.text(0, 0.5, "Not enough festival accidents per province to compute a reliable lift.", color=INK_2)
        return _save(fig, out_dir, "06_province_lift.png")
    return _hbar(top, f"Top {n} provinces where festivals hit hardest",
                 "Festival accidents per day ÷ normal accidents per day (1 = no change)",
                 COLORS["Festival"], "{:.1f}×", out_dir, "06_province_lift.png", ref=1.0)


def dumbbell(table: pd.DataFrame, title: str, out_dir: Path, name: str) -> Path:
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.3))
    y = range(len(t))
    normal, fest = 100 * t["share_Normal days"], 100 * t["share_Festival"]
    ax.hlines(y, normal, fest, color=GRID, lw=3)
    ax.scatter(normal, y, s=60, color=COLORS["Normal days"], label="Normal days", zorder=3, edgecolor=SURFACE, lw=2)
    ax.scatter(fest, y, s=60, color=COLORS["Festival"], label="Festival", zorder=3, edgecolor=SURFACE, lw=2)
    for yi, f, chg in zip(y, fest, t["share_change_pts"]):
        ax.annotate(f"{chg:+.1f} pts", (max(f, normal.iloc[yi]), yi), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_yticks(list(y), t.index)
    ax.set_xlabel("% of accidents")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.margins(x=0.15)
    ax.legend(loc="lower right")
    return _save(fig, out_dir, name)


def yearly(by_year: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for fest, g in by_year.groupby("period"):
        ax.plot(g["festival_year"], g["deaths"], marker="o", lw=2, ms=7, color=COLORS[fest], label=fest)
        last = g.iloc[-1]
        ax.annotate(f"{fest} {int(last['festival_year'])}: {last['deaths']:,.0f}", (last["festival_year"], last["deaths"]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xticks(sorted(by_year["festival_year"].unique()))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Deaths in the 7-day window")
    ax.set_title("Festival deaths by year")
    ax.margins(x=0.2)
    ax.legend(loc="lower left")
    return _save(fig, out_dir, "09_festival_deaths_by_year.png")


def fatal_map(df: pd.DataFrame, out_dir: Path) -> Path:
    pts = df.dropna(subset=["latitude", "longitude"])
    fest_fatal = pts[(pts["period"] != "Normal days") & pts["is_fatal"]]
    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.scatter(pts["longitude"], pts["latitude"], s=1, color=GRID, alpha=0.6, label="All accidents", rasterized=True)
    ax.scatter(fest_fatal["longitude"], fest_fatal["latitude"], s=9, color=COLORS["Festival"], alpha=0.8,
               edgecolor="none", label="Fatal accidents during festivals")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.set_title("Where festival deaths happen")
    ax.legend(loc="lower left", markerscale=1.5)
    return _save(fig, out_dir, "10_fatal_map.png")
