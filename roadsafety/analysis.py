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
    """Accidents and deaths per day in each period, and the ratio to normal days."""
    days = calendar_days(df)["period"].value_counts().rename("days")
    table = df.groupby("period").apply(_rates, include_groups=False).join(days)
    table["accidents_per_day"] = table["accidents"] / table["days"]
    table["deaths_per_day"] = table["deaths"] / table["days"]
    normal = table.loc["Normal days"]
    table["accidents_vs_normal"] = table["accidents_per_day"] / normal["accidents_per_day"]
    table["deaths_vs_normal"] = table["deaths_per_day"] / normal["deaths_per_day"]
    return table.reindex([p for p in PERIODS if p in table.index])


def _complete_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Campaign windows that are not cut off by the start or end of the data."""
    days = calendar_days(df)
    return days[days["complete_window"]].drop_duplicates(["period", "festival_year"])[["period", "festival_year"]]


def campaign_windows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per detected campaign: first and last day, length, and totals."""
    days = calendar_days(df).dropna(subset=["festival_year"])
    win = days.groupby(["period", "festival_year"]).agg(
        start=("date", "min"), end=("date", "max"), days=("date", "size"), complete=("complete_window", "all")
    )
    fest = df.dropna(subset=["festival_year"])
    if fest.empty:
        cols = ["period", "festival_year", "start", "end", "days", "complete", "accidents", "deaths",
                "deaths_per_day", "accidents_per_day", "deaths_per_100_accidents"]
        return pd.DataFrame(columns=cols)
    totals = fest.groupby(["period", "festival_year"]).apply(_rates, include_groups=False)
    out = win.join(totals)
    out["deaths_per_day"] = out["deaths"] / out["days"]
    out["accidents_per_day"] = out["accidents"] / out["days"]
    return out.reset_index().sort_values("start")


def festival_by_year(df: pd.DataFrame) -> pd.DataFrame:
    w = campaign_windows(df)
    return w[w["complete"]].reset_index(drop=True)


def around_nominal_start(df: pd.DataFrame, before: int = 10, after: int = 17) -> pd.DataFrame:
    """Average accidents per day by offset from the nominal campaign start (29 Dec / 11 Apr).

    Uses fixed calendar dates, not the detected windows, so it is independent
    evidence of the step in recording when a campaign starts.
    """
    daily = df.groupby("date").size()
    daily = daily.reindex(pd.date_range(df["date"].min(), df["date"].max()), fill_value=0)
    rows = []
    for year in range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1):
        for fest, start in [("New Year", pd.Timestamp(year, 12, 29)), ("Songkran", pd.Timestamp(year, 4, 11))]:
            span = pd.date_range(start - pd.Timedelta(days=before), start + pd.Timedelta(days=after))
            if span[0] < daily.index[0] or span[-1] > daily.index[-1]:
                continue
            rows.append(pd.DataFrame({"period": fest, "offset": range(-before, after + 1), "accidents": daily[span].to_numpy()}))
    return pd.concat(rows).groupby(["period", "offset"])["accidents"].mean().reset_index()


def festival_day_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average accidents and deaths on each calendar day of the campaigns.

    Keyed by calendar date (e.g. 31 Dec, 13 Apr) because campaigns start on
    different days each year; only dates that were campaign days in at least
    half of the complete windows are kept.
    """
    windows = _complete_windows(df)
    fest = df.merge(windows, on=["period", "festival_year"])
    days = calendar_days(df).merge(windows, on=["period", "festival_year"])
    days["md"] = days["date"].dt.strftime("%m-%d")
    fest = fest.assign(md=fest["date"].dt.strftime("%m-%d"))
    n_days = days.groupby(["period", "md"]).size().rename("n_years")
    daily = fest.groupby(["period", "md"]).agg(accidents=("date", "size"), deaths=("fatalities", "sum")).join(n_days)
    n_windows = windows.groupby("period").size()
    daily = daily[daily["n_years"] >= daily.index.get_level_values("period").map(n_windows) / 2]
    daily["accidents_per_day"] = daily["accidents"] / daily["n_years"]
    daily["deaths_per_day"] = daily["deaths"] / daily["n_years"]
    daily = daily.reset_index()
    # Order New Year days across the turn of the year (Dec before Jan).
    daily["order"] = daily["md"].where(~daily["md"].str.startswith("01"), "13" + daily["md"].str[2:])
    return daily.sort_values(["period", "order"]).drop(columns="order").reset_index(drop=True)


def hour_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Share of accidents and deaths per 100 accidents by hour, festival vs normal days."""
    d = df[df["hour"].notna()].assign(group=lambda x: x["period"].where(x["period"] == "Normal days", "Festival"))
    table = d.groupby(["group", "hour"]).apply(_rates, include_groups=False).reset_index()
    table["share_of_accidents"] = table["accidents"] / table.groupby("group")["accidents"].transform("sum")
    return table


def province_table(df: pd.DataFrame, min_festival_accidents: int = 100) -> pd.DataFrame:
    """Per province: festival deaths (burden) and deaths per 100 festival accidents (severity).

    Severity is left blank for provinces with fewer than `min_festival_accidents`
    festival accidents, where one crash moves the rate too much.
    """
    d = df[df["province_th"].notna()]
    fest = d[d["period"] != "Normal days"]
    g = fest.groupby("province_th").agg(
        festival_accidents=("date", "size"), festival_deaths=("fatalities", "sum")
    )
    g["share_of_festival_deaths"] = g["festival_deaths"] / g["festival_deaths"].sum()
    g["festival_deaths_per_100_accidents"] = 100 * g["festival_deaths"] / g["festival_accidents"]
    g.loc[g["festival_accidents"] < min_festival_accidents, "festival_deaths_per_100_accidents"] = float("nan")
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


def agency_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Share of recorded accidents by reporting agency, campaign vs normal days."""
    group = df["period"].where(df["period"] == "Normal days", "Festival")
    return pd.crosstab(df["agency"], group, normalize="columns")


def region_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["region"].notna()]
    fest = d[d["period"] != "Normal days"]
    if fest.empty:
        return pd.DataFrame(columns=["accidents", "deaths", "injuries", "fatal_accidents", "deaths_per_100_accidents"])
    return fest.groupby("region").apply(_rates, include_groups=False).sort_values("deaths", ascending=False)
