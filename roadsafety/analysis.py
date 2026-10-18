"""Aggregations behind every chart and number in the report.

All rates are per calendar day (see features.calendar_days), and "Festival"
means the two seven-day campaign windows combined.
"""

from __future__ import annotations

import pandas as pd

from .features import PERIODS, calendar_days


def _rates(group: pd.DataFrame) -> pd.Series:
    acc = len(group)
    deaths = group["fatalities"].sum()
    return pd.Series(
        {
            "accidents": acc,
            "deaths": deaths,
            "injuries": group["injuries"].sum(),
            "fatal_accidents": group["is_fatal"].sum(),
            "deaths_per_100_accidents": 100 * deaths / acc if acc else float("nan"),
        }
    )


def overview(df: pd.DataFrame) -> dict:
    return {
        "accidents": len(df),
        "deaths": int(df["fatalities"].sum()),
        "injuries": int(df["injuries"].sum()),
        "first_date": df["date"].min().date(),
        "last_date": df["date"].max().date(),
        "years": sorted(df["year"].unique().tolist()),
        "provinces": int(df["province_th"].nunique()),
    }


def period_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Accidents and deaths per day in each period, and the lift over normal days."""
    days = calendar_days(df)["period"].value_counts().rename("days")
    table = df.groupby("period").apply(_rates, include_groups=False).join(days)
    table["accidents_per_day"] = table["accidents"] / table["days"]
    table["deaths_per_day"] = table["deaths"] / table["days"]
    normal = table.loc["Normal days"]
    table["accident_lift"] = table["accidents_per_day"] / normal["accidents_per_day"]
    table["death_lift"] = table["deaths_per_day"] / normal["deaths_per_day"]
    return table.reindex([p for p in PERIODS if p in table.index])


def _complete_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Festival windows whose seven days all fall inside the data's date range."""
    days = calendar_days(df)
    sizes = days.dropna(subset=["festival_year"]).groupby(["period", "festival_year"]).size()
    return sizes[sizes == 7].reset_index()[["period", "festival_year"]]


def festival_by_year(df: pd.DataFrame) -> pd.DataFrame:
    windows = _complete_windows(df)
    fest = df.merge(windows, on=["period", "festival_year"])
    table = fest.groupby(["period", "festival_year"]).apply(_rates, include_groups=False)
    return table.reset_index()


def festival_day_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average accidents and deaths on day 1..7 of each festival."""
    windows = _complete_windows(df)
    fest = df.merge(windows, on=["period", "festival_year"])
    n_years = windows.groupby("period").size().rename("n_windows")
    daily = fest.groupby(["period", "festival_day"]).agg(accidents=("date", "size"), deaths=("fatalities", "sum"))
    daily = daily.join(n_years, on="period")
    daily["accidents_per_day"] = daily["accidents"] / daily["n_windows"]
    daily["deaths_per_day"] = daily["deaths"] / daily["n_windows"]
    return daily.reset_index()


def hour_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Share of accidents and deaths per 100 accidents by hour, festival vs normal days."""
    d = df[df["hour"].notna()].assign(group=lambda x: x["period"].where(x["period"] == "Normal days", "Festival"))
    table = d.groupby(["group", "hour"]).apply(_rates, include_groups=False).reset_index()
    table["share_of_accidents"] = table["accidents"] / table.groupby("group")["accidents"].transform("sum")
    return table


def weekday_hour(df: pd.DataFrame, value: str = "accidents") -> pd.DataFrame:
    d = df[df["hour"].notna()]
    if value == "deaths":
        m = d.pivot_table(index="weekday", columns="hour", values="fatalities", aggfunc="sum", observed=False)
    else:
        m = d.pivot_table(index="weekday", columns="hour", values="date", aggfunc="size", observed=False)
    return m.reindex(columns=range(24)).fillna(0)


def province_table(df: pd.DataFrame, min_festival_accidents: int = 20) -> pd.DataFrame:
    """Per province: festival totals and the festival-vs-normal daily lift.

    The lift compares a province with itself, so it does not need population or
    road length to be fair across provinces. Provinces with fewer than
    `min_festival_accidents` festival accidents get no lift (too noisy).
    """
    days = calendar_days(df)
    fest_days = (days["period"] != "Normal days").sum()
    normal_days = (days["period"] == "Normal days").sum()
    d = df[df["province_th"].notna()]
    is_fest = d["period"] != "Normal days"
    g = pd.DataFrame(
        {
            "festival_accidents": d[is_fest].groupby("province_th").size(),
            "festival_deaths": d[is_fest].groupby("province_th")["fatalities"].sum(),
            "normal_accidents": d[~is_fest].groupby("province_th").size(),
            "normal_deaths": d[~is_fest].groupby("province_th")["fatalities"].sum(),
        }
    ).fillna(0)
    g["festival_accidents_per_day"] = g["festival_accidents"] / fest_days
    g["normal_accidents_per_day"] = g["normal_accidents"] / normal_days
    g["lift"] = g["festival_accidents_per_day"] / g["normal_accidents_per_day"]
    g.loc[g["festival_accidents"] < min_festival_accidents, "lift"] = float("nan")
    g["festival_deaths_per_100_accidents"] = 100 * g["festival_deaths"] / g["festival_accidents"].where(g["festival_accidents"] > 0)
    names = df.drop_duplicates("province_th").set_index("province_th")[["province_en", "region"]]
    return g.join(names).sort_values("festival_deaths", ascending=False)


def category_comparison(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Share of accidents per category on festival vs normal days, plus severity."""
    d = df.assign(group=df["period"].where(df["period"] == "Normal days", "Festival"))
    share = pd.crosstab(d[column], d["group"], normalize="columns")
    sev = d.groupby(column).apply(_rates, include_groups=False)[["accidents", "deaths", "deaths_per_100_accidents"]]
    out = share.add_prefix("share_").join(sev)
    out["share_change_pts"] = 100 * (out.get("share_Festival", 0) - out.get("share_Normal days", 0))
    return out.sort_values("share_Festival" if "share_Festival" in out else "accidents", ascending=False)


def region_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["region"].notna()]
    fest = d[d["period"] != "Normal days"]
    return fest.groupby("region").apply(_rates, include_groups=False).sort_values("deaths", ascending=False)
