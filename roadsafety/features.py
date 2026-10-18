"""Calendar features, with the festival windows used by Thailand's road-safety campaigns.

The government's "7 dangerous days" (7 วันอันตราย) campaigns run around

* New Year: nominally 29 December - 4 January
* Songkran: nominally 11 - 17 April

but the exact days move from year to year (Songkran 2021 started on 9 April,
New Year 2025 on 27 December, Songkran 2020 was cancelled during COVID-19).
During a campaign the agencies record accidents far more intensively, so the
daily count steps up three- to four-fold on the first campaign day and drops
back the day after it ends. `detect_campaign_days` finds those steps in the
data itself, so each year's real campaign days are labelled, not a fixed guess.

A New Year window is labelled by the year it ends in, so
29 Dec 2022 - 4 Jan 2023 is "New Year 2023".
"""

from __future__ import annotations

import pandas as pd

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FESTIVALS = ["New Year", "Songkran"]
PERIODS = FESTIVALS + ["Normal days"]


def _season(dates: pd.Series) -> pd.Series:
    """Which festival season a date could belong to (Dec 15 - Jan 15, April), else NA."""
    md = dates.dt.month * 100 + dates.dt.day
    season = pd.Series(pd.NA, index=dates.index, dtype="string")
    season[(md >= 1215) | (md <= 115)] = "New Year"
    season[dates.dt.month == 4] = "Songkran"
    return season


def _festival_year(dates: pd.Series) -> pd.Series:
    return (dates.dt.year + (dates.dt.month == 12).astype(int)).astype("Int64")


def fixed_windows(dates: pd.Series) -> pd.Series:
    """True on the nominal campaign days (29 Dec - 4 Jan, 11 - 17 Apr)."""
    month, day = dates.dt.month, dates.dt.day
    return ((month == 4) & day.between(11, 17)) | ((month == 12) & (day >= 29)) | ((month == 1) & (day <= 4))


def detect_campaign_days(dates: pd.Series, factor: float = 1.8, window: int = 29, min_days: int = 4) -> pd.Series:
    """Flag the calendar days that fall inside a campaign, from the step in daily counts.

    A day is a campaign day when it is in a festival season, its accident count
    is at least `factor` times the rolling `window`-day median, and it belongs to
    a run of at least `min_days` such days in that season (one-off spikes are not
    campaigns). Returns a boolean Series indexed by every calendar day.
    """
    days = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = dates.value_counts().reindex(days, fill_value=0)
    baseline = daily.rolling(window, center=True, min_periods=7).median()
    season = _season(pd.Series(days, index=days))
    high = (daily >= factor * baseline) & season.notna()
    key = season.fillna("") + _festival_year(pd.Series(days, index=days)).astype(str)
    run_len = high.groupby(key).transform("sum")
    return high & (run_len >= min_days)


def festival_of(dates: pd.Series, campaign: pd.Series | None = None) -> pd.DataFrame:
    """Festival name, festival year and day-of-campaign (1, 2, ...) for each date.

    `campaign` is a boolean Series indexed by calendar day (see detect_campaign_days);
    without it the nominal fixed windows are used.
    """
    is_campaign = dates.map(campaign).fillna(False).astype(bool) if campaign is not None else fixed_windows(dates)
    season = _season(dates)
    festival = season.where(is_campaign, "Normal days").fillna("Normal days")
    fest_year = _festival_year(dates).where(is_campaign)

    fest_day = pd.Series(pd.NA, index=dates.index, dtype="Int64")
    if campaign is None:
        nominal_start = pd.to_datetime(
            pd.DataFrame({"year": dates.dt.year - (dates.dt.month == 1), "month": 12, "day": 29})
        ).where(season == "New Year", pd.to_datetime(pd.DataFrame({"year": dates.dt.year, "month": 4, "day": 11})))
        fest_day[is_campaign] = (dates[is_campaign] - nominal_start[is_campaign]).dt.days + 1
    elif is_campaign.any():
        start = dates[is_campaign].groupby([festival[is_campaign], fest_year[is_campaign]]).transform("min")
        fest_day[is_campaign] = (dates[is_campaign] - start).dt.days + 1
    return pd.DataFrame({"period": festival.astype("string"), "festival_year": fest_year, "festival_day": fest_day})


def add_features(df: pd.DataFrame, detect_campaigns: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["weekday"] = pd.Categorical(out["date"].dt.dayofweek.map(dict(enumerate(WEEKDAYS))), WEEKDAYS, ordered=True)
    out["is_weekend"] = out["date"].dt.dayofweek >= 5
    campaign = detect_campaign_days(out["date"]) if detect_campaigns else None
    out = out.join(festival_of(out["date"], campaign))
    out["is_fatal"] = out["fatalities"] > 0
    out["casualties"] = out["fatalities"] + out["injuries"]
    out.attrs["campaign_days"] = campaign
    return out


def calendar_days(df: pd.DataFrame) -> pd.DataFrame:
    """Every calendar day between the first and last incident, tagged with its period.

    Rates must be divided by calendar days, not by days that happen to have an
    incident in the file, otherwise quiet days vanish and normal-day rates are inflated.
    """
    days = pd.Series(pd.date_range(df["date"].min(), df["date"].max(), freq="D"))
    out = pd.DataFrame({"date": days}).join(festival_of(days, df.attrs.get("campaign_days")))
    # A window cut by the start or end of the data is not a complete campaign.
    first, last = days.iloc[0], days.iloc[-1]
    win = out.dropna(subset=["festival_year"]).groupby(["period", "festival_year"])["date"].agg(["min", "max"])
    partial = win[(win["min"] == first) | (win["max"] == last)].index
    out["complete_window"] = [
        pd.notna(y) and (p, y) not in partial for p, y in zip(out["period"], out["festival_year"])
    ]
    return out
