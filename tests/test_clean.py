from pathlib import Path

import pandas as pd
import pytest

from roadsafety.clean import CleaningLog, clean, parse_date, parse_time, read_raw
from roadsafety.provinces import PROVINCE_TABLE, normalize_province

FIXTURES = Path(__file__).parent / "fixtures"


def test_province_table_is_complete():
    assert len(PROVINCE_TABLE) == 77
    assert PROVINCE_TABLE["province_th"].is_unique
    assert PROVINCE_TABLE["region"].value_counts().to_dict() == {
        "Central": 22, "Northeast": 20, "South": 14, "North": 9, "East": 7, "West": 5,
    }


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("จ.เชียงใหม่", "เชียงใหม่"),
        ("จังหวัด นครราชสีมา", "นครราชสีมา"),
        ("โคราช", "นครราชสีมา"),
        ("กทม.", "กรุงเทพมหานคร"),
        (" ชลบุรี ", "ชลบุรี"),
        ("Chon Buri", "ชลบุรี"),
        ("CHONBURI", "ชลบุรี"),
        ("หนองบัวลําภู", "หนองบัวลำภู"),
        ("Phra Nakhon Si Ayutthaya Province", "พระนครศรีอยุธยา"),
        ("เมืองลับแล", None),
    ],
)
def test_normalize_province(raw, expected):
    out = normalize_province(pd.Series([raw])).iloc[0]
    assert (out if pd.notna(out) else None) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("13/04/2566", "2023-04-13"),
        ("13/04/2023", "2023-04-13"),
        ("๑๔/๐๔/๒๕๖๖", "2023-04-14"),
        ("13 เม.ย. 2566", "2023-04-13"),
        ("1 มกราคม 2567", "2024-01-01"),
        ("2023-01-02", "2023-01-02"),
        ("2023-01-02 23:10:00", "2023-01-02"),
        ("2/1/2566 08:00", "2023-01-02"),
        ("45000", "2023-03-15"),
        ("ไม่ระบุ", None),
    ],
)
def test_parse_date(raw, expected):
    out = parse_date(pd.Series([raw])).iloc[0]
    assert (out.strftime("%Y-%m-%d") if pd.notna(out) else None) == expected


@pytest.mark.parametrize(
    "raw, minutes",
    [("14:30", 870), ("21.45 น.", 1305), ("0230", 150), ("930", 570), ("14:30:00", 870), ("8", 480),
     ("25:99", None), (None, None)],
)
def test_parse_time(raw, minutes):
    out = parse_time(pd.Series([raw], dtype="string")).iloc[0]
    assert (out.total_seconds() / 60 if pd.notna(out) else None) == minutes


def test_clean_thai_file():
    log = CleaningLog()
    df = clean(read_raw(FIXTURES / "messy_thai_headers.csv"), log)
    by_code = df.set_index("acc_code")

    assert "A006" not in by_code.index  # unparseable date dropped
    assert by_code.index.is_unique  # duplicate A001 removed
    assert len(df) == 7

    assert by_code.loc["A002", "province_th"] == "นครราชสีมา"
    assert by_code.loc["A003", "province_en"] == "Bangkok"
    assert pd.isna(by_code.loc["A008", "province_th"])  # a district, not a province

    # lat/lon swapped in the raw file
    assert by_code.loc["A002", ["latitude", "longitude"]].tolist() == [14.97, 102.10]
    assert pd.isna(by_code.loc["A005", "latitude"])  # 0,0 is outside Thailand

    assert by_code.loc["A002", "fatalities"] == 2  # Thai digit
    assert by_code.loc["A001", "injuries"] == 2  # serious + minor
    assert by_code.loc["A003", "injuries"] == 1  # blank serious counted as 0
    assert pd.isna(by_code.loc["A005", "hour"])  # 25:99

    assert by_code.loc["A001", "cause_group"] == "Speeding"
    assert by_code.loc["A002", "cause_group"] == "Drunk driving"
    assert by_code.loc["A003", "cause_group"] == "Drowsy / fatigue"
    assert by_code.loc["A004", "cause_group"] == "Cutting in"
    assert by_code.loc["A005", "cause_group"] == "Unknown"
    assert by_code.loc["A002", "vehicle_group"] == "Pickup"
    assert by_code.loc["A004", "vehicle_group"] == "Truck"
    assert by_code.loc["A003", "vehicle_group"] == "Car"

    steps = log.to_frame().set_index("step")["rows_affected"]
    assert steps["duplicate rows removed"] == 1
    assert steps["latitude/longitude swapped back"] == 1


def test_clean_english_file():
    df = clean(read_raw(FIXTURES / "english_headers.csv"))
    assert df["province_en"].tolist() == ["Khon Kaen", "Phuket"]
    assert df["hour"].tolist() == [18, 23]
    assert df["cause_group"].tolist() == ["Speeding", "Drunk driving"]
    assert df["vehicle_group"].tolist() == ["Motorcycle", "Car"]


def test_missing_columns_has_helpful_error():
    with pytest.raises(ValueError, match="Headers present"):
        clean(pd.DataFrame({"foo": ["1"]}))


def test_month_first_files_are_detected():
    us = pd.Series(["7/15/2020", "1/2/2020", "12/31/2020"])
    assert parse_date(us).dt.strftime("%Y-%m-%d").tolist() == ["2020-07-15", "2020-01-02", "2020-12-31"]
    th = pd.Series(["15/7/2020", "1/2/2020"])
    assert parse_date(th).dt.strftime("%Y-%m-%d").tolist() == ["2020-07-15", "2020-02-01"]


def test_excel_day_fraction_times():
    mostly_fractions = pd.Series(["0.340277778", "0.25", "0", "0.5"], dtype="string")
    assert [t.total_seconds() / 60 for t in parse_time(mostly_fractions)] == [490, 360, 0, 720]
    normal = pd.Series(["0.30", "14:00", "9:15"], dtype="string")
    assert [t.total_seconds() / 60 for t in parse_time(normal)] == [30, 840, 555]


def test_swapped_duplicate_keeps_copy_matching_neighbours():
    codes = [str(c) for c in range(100, 140)]
    dates = [f"1/{d}/2023" for d in range(1, 21)] + [f"2/{d}/2023" for d in range(1, 21)]  # month/day
    raw = pd.DataFrame({"ACC_CODE": codes, "วันที่เกิดเหตุ": dates, "จังหวัด": "ขอนแก่น"})
    # code 110 sits among early-January codes; its second copy has day and month swapped.
    raw = pd.concat([raw, pd.DataFrame({"ACC_CODE": ["110"], "วันที่เกิดเหตุ": ["11/1/2023"], "จังหวัด": ["ขอนแก่น"]})])
    log = CleaningLog()
    df = clean(raw, log).set_index("acc_code")
    assert df.loc["110", "date"] == pd.Timestamp("2023-01-11")
    assert len(df) == 40


def test_incident_date_swapped_against_report_date():
    raw = pd.DataFrame({
        "วันที่เกิดเหตุ": ["1/15/2022", "02/01/2022", "02/01/2022"],  # file is month/day
        "วันที่รายงาน": ["1/16/2022", "01/03/2022", "02/03/2022"],
        "จังหวัด": "ขอนแก่น",
    })
    df = clean(raw)
    # Row 2: read as Feb 1 it would be reported on Jan 3, before it happened -> really Jan 2.
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2022-01-15", "2022-01-02", "2022-02-01"]
