"""Calendar features, with the festival windows used by Thailand's road-safety campaigns.

The government's "7 dangerous days" (7 วันอันตราย) campaigns count accidents over
a fixed seven-day window each festival:

* New Year: 29 December - 4 January
* Songkran: 11 - 17 April

Those windows are used here so the numbers line up with the official campaign
reporting. A New Year window is labelled by the year it ends in, so
29 Dec 2022 - 4 Jan 2023 is "New Year 2023".
"""

from __future__ import annotations

import pandas as pd

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FESTIVALS = ["New Year", "Songkran"]
PERIODS = FESTIVALS + ["Normal days"]


def festival_of(dates: pd.Series) -> pd.DataFrame:
    """Return festival name, festival year and day-of-festival (1-7) for each date."""
    month, day, year = dates.dt.month, dates.dt.day, dates.dt.year
    songkran = (month == 4) & day.between(11, 17)
    ny_dec = (month == 12) & (day >= 29)
    ny_jan = (month == 1) & (day <= 4)

    festival = pd.Series("Normal days", index=dates.index, dtype="string")
    festival[songkran] = "Songkran"
    festival[ny_dec | ny_jan] = "New Year"

    fest_year = pd.Series(pd.NA, index=dates.index, dtype="Int64")
    fest_year[songkran | ny_jan] = year[songkran | ny_jan]
    fest_year[ny_dec] = year[ny_dec] + 1

    fest_day = pd.Series(pd.NA, index=dates.index, dtype="Int64")
    fest_day[songkran] = day[songkran] - 10
    fest_day[ny_dec] = day[ny_dec] - 28
    fest_day[ny_jan] = day[ny_jan] + 3
    return pd.DataFrame({"period": festival, "festival_year": fest_year, "festival_day": fest_day})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["weekday"] = pd.Categorical(out["date"].dt.dayofweek.map(dict(enumerate(WEEKDAYS))), WEEKDAYS, ordered=True)
    out["is_weekend"] = out["date"].dt.dayofweek >= 5
    out = out.join(festival_of(out["date"]))
    out["is_fatal"] = out["fatalities"] > 0
    out["casualties"] = out["fatalities"] + out["injuries"]
    return out


def calendar_days(df: pd.DataFrame) -> pd.DataFrame:
    """Every calendar day between the first and last incident, tagged with its period.

    Rates must be divided by calendar days, not by days that happen to have an
    incident in the file, otherwise quiet days vanish and normal-day rates are inflated.
    """
    days = pd.Series(pd.date_range(df["date"].min(), df["date"].max(), freq="D"))
    return pd.DataFrame({"date": days}).join(festival_of(days))
