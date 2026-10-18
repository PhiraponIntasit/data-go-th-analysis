import numpy as np
import pandas as pd
import pytest

from roadsafety import analysis
from roadsafety.features import add_features, festival_of
from roadsafety.provinces import PROVINCE_TABLE
from roadsafety.report import build


@pytest.mark.parametrize(
    "date, period, year, day",
    [
        ("2022-12-29", "New Year", 2023, 1),
        ("2023-01-01", "New Year", 2023, 4),
        ("2023-01-04", "New Year", 2023, 7),
        ("2023-01-05", "Normal days", None, None),
        ("2023-04-11", "Songkran", 2023, 1),
        ("2023-04-17", "Songkran", 2023, 7),
        ("2023-04-18", "Normal days", None, None),
    ],
)
def test_festival_windows(date, period, year, day):
    row = festival_of(pd.Series(pd.to_datetime([date]))).iloc[0]
    assert row["period"] == period
    assert (None if pd.isna(row["festival_year"]) else row["festival_year"]) == year
    assert (None if pd.isna(row["festival_day"]) else row["festival_day"]) == day


def _tiny() -> pd.DataFrame:
    # Two accidents on each of the 14 festival days of 2023, one on each normal day.
    days = pd.date_range("2022-12-29", "2023-04-17", freq="D")
    fest = festival_of(pd.Series(days))["period"] != "Normal days"
    dates = np.repeat(days, np.where(fest, 2, 1))
    n = len(dates)
    return add_features(pd.DataFrame({
        "date": dates, "hour": pd.array([20] * n, dtype="Int64"), "fatalities": 1, "injuries": 0,
        "province_th": "ขอนแก่น", "province_en": "Khon Kaen", "region": "Northeast",
        "cause_group": "Speeding", "vehicle_group": "Motorcycle", "latitude": np.nan, "longitude": np.nan,
    }))


def test_period_rates_use_calendar_days():
    rates = analysis.period_rates(_tiny())
    assert rates.loc["Songkran", "days"] == 7
    assert rates.loc["New Year", "accidents_per_day"] == 2
    assert rates.loc["Normal days", "accidents_per_day"] == 1
    assert rates.loc["Songkran", "accident_lift"] == 2


def test_festival_day_profile_only_complete_windows():
    df = _tiny()
    partial = df[df["date"] >= "2022-12-31"]  # New Year window cut short
    prof = analysis.festival_day_profile(partial)
    assert set(prof["period"]) == {"Songkran"}


def test_province_lift():
    prov = analysis.province_table(_tiny(), min_festival_accidents=1)
    assert prov.loc["ขอนแก่น", "lift"] == pytest.approx(2.0)


def test_end_to_end_report(tmp_path):
    """Smoke test on random synthetic rows: the pipeline runs and writes every figure."""
    rng = np.random.default_rng(0)
    n = 3000
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    prov = PROVINCE_TABLE.sample(n, replace=True, random_state=0)
    raw = pd.DataFrame({
        "ACC_CODE": [f"X{i}" for i in range(n)],
        "วันที่เกิดเหตุ": dates.strftime("%d/%m/") + (dates.year + 543).astype(str),
        "เวลา": [f"{h:02d}:{m:02d}" for h, m in zip(rng.integers(0, 24, n), rng.integers(0, 60, n))],
        "จังหวัด": prov["province_th"].to_numpy(),
        "รถคันที่1": rng.choice(["รถจักรยานยนต์", "รถเก๋ง", "รถบรรทุก 10 ล้อ"], n),
        "มูลเหตุสันนิษฐาน": rng.choice(["ขับรถเร็ว", "เมาสุรา", "หลับใน", "-"], n),
        "LATITUDE": rng.uniform(6, 20, n).round(4),
        "LONGITUDE": rng.uniform(98, 105, n).round(4),
        "จำนวนผู้เสียชีวิต": rng.poisson(0.3, n),
        "รวมจำนวนผู้บาดเจ็บ": rng.poisson(1, n),
    })
    path = tmp_path / "synthetic.csv"
    raw.to_csv(path, index=False)
    report = build([path], out_dir=tmp_path / "out", processed_dir=tmp_path / "processed")
    text = report.read_text(encoding="utf-8")
    assert "## 5. Policy recommendations" in text
    assert len(list((tmp_path / "out" / "figures").glob("*.png"))) == 10
