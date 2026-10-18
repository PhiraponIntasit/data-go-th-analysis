"""Turn raw Department of Highways accident files into one tidy, typed table.

The raw files on data.go.th are published one per year, and they are not
consistent with each other: Thai column headers in some years and English in
others, Buddhist-Era years, Thai digits, Thai month abbreviations, times typed
as "14.30 น.", province names with or without the "จ." prefix, latitude and
longitude swapped, and duplicated rows. Every rule below exists because of one
of those problems, and each step writes what it did to a cleaning log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .provinces import PROVINCE_TABLE, normalize_province

# Canonical column -> header spellings seen across years (compared after _norm_header).
COLUMN_ALIASES: dict[str, list[str]] = {
    "acc_code": ["acc_code", "acccode", "รหัสอุบัติเหตุ"],
    "incident_datetime": ["incident_datetime", "วันที่และเวลาที่เกิดเหตุ"],
    "incident_date": ["วันที่เกิดเหตุ", "incident_date", "date"],
    "incident_time": ["เวลา", "เวลาที่เกิดเหตุ", "incident_time", "time"],
    "province": ["จังหวัด", "province_th", "province", "province_en"],
    "agency": ["หน่วยงาน", "agency"],
    "route": ["สายทาง", "route", "ชื่อสายทาง"],
    "vehicle": ["รถคันที่1", "รถคันที่ 1", "vehicle_type", "ประเภทรถ"],
    "cause": ["มูลเหตุสันนิษฐาน", "presumed_cause", "สาเหตุ"],
    "accident_type": ["ลักษณะการเกิดอุบัติเหตุ", "accident_type", "ลักษณะการเกิดเหตุ"],
    "weather": ["สภาพอากาศ", "weather_condition", "weather"],
    "road_description": ["บริเวณที่เกิดเหตุ/ลักษณะทาง", "บริเวณที่เกิดเหตุ", "road_description"],
    "latitude": ["latitude", "lat", "ละติจูด"],
    "longitude": ["longitude", "lon", "long", "ลองจิจูด"],
    "vehicles_involved": ["จำนวนรถที่เกิดเหตุ", "number_of_vehicles_involved"],
    "fatalities": ["จำนวนผู้เสียชีวิต", "number_of_fatalities", "ผู้เสียชีวิต", "เสียชีวิต"],
    "injuries": ["รวมจำนวนผู้บาดเจ็บ", "number_of_injuries", "จำนวนผู้บาดเจ็บ", "ผู้บาดเจ็บ"],
    "injuries_serious": ["จำนวนผู้บาดเจ็บสาหัส", "บาดเจ็บสาหัส"],
    "injuries_minor": ["จำนวนผู้บาดเจ็บเล็กน้อย", "บาดเจ็บเล็กน้อย"],
}

# Province names are in Thai in this table; keep the English form if a file only has that.
REQUIRED = ["province"]

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

THAI_MONTHS = {
    "มกราคม": 1, "ม.ค.": 1, "มค": 1,
    "กุมภาพันธ์": 2, "ก.พ.": 2, "กพ": 2,
    "มีนาคม": 3, "มี.ค.": 3, "มีค": 3,
    "เมษายน": 4, "เม.ย.": 4, "เมย": 4,
    "พฤษภาคม": 5, "พ.ค.": 5, "พค": 5,
    "มิถุนายน": 6, "มิ.ย.": 6, "มิย": 6,
    "กรกฎาคม": 7, "ก.ค.": 7, "กค": 7,
    "สิงหาคม": 8, "ส.ค.": 8, "สค": 8,
    "กันยายน": 9, "ก.ย.": 9, "กย": 9,
    "ตุลาคม": 10, "ต.ค.": 10, "ตค": 10,
    "พฤศจิกายน": 11, "พ.ย.": 11, "พย": 11,
    "ธันวาคม": 12, "ธ.ค.": 12, "ธค": 12,
}

# Keyword rules: first match wins, so the more specific causes come first.
CAUSE_RULES: list[tuple[str, list[str]]] = [
    ("Drunk driving", ["เมา", "แอลกอฮอล์", "สุรา", "drunk", "alcohol"]),
    ("Drowsy / fatigue", ["หลับใน", "ง่วง", "อ่อนเพลีย", "drows", "fatigue", "asleep"]),
    ("Phone / distraction", ["โทรศัพท์", "phone", "distract"]),
    ("Speeding", ["เร็ว", "speed"]),
    ("Cutting in", ["ตัดหน้า", "cut in", "cutting"]),
    ("Unsafe overtaking", ["แซง", "overtak"]),
    ("Signal / sign violation", ["สัญญาณ", "ฝ่าฝืน", "ป้าย", "signal", "red light", "sign"]),
    ("Following too closely", ["กระชั้นชิด", "ระยะ", "tailgat", "closely"]),
    ("Vehicle defect", ["อุปกรณ์", "เบรก", "ยาง", "ชำรุด", "บรรทุกเกิน", "defect", "brake", "tire", "tyre", "overload"]),
    ("Road / environment", ["ถนน", "ทาง", "แสงสว่าง", "ไฟส่อง", "สัตว์", "ลื่น", "road", "light", "animal", "slippery"]),
]

VEHICLE_RULES: list[tuple[str, list[str]]] = [
    ("Motorcycle", ["จักรยานยนต์", "จยย", "motorcycle", "motorbike"]),
    ("Pickup", ["ปิคอัพ", "ปิกอัพ", "กระบะ", "pickup", "pick-up"]),
    ("Truck", ["บรรทุก", "พ่วง", "truck", "trailer"]),
    ("Bus / van", ["โดยสาร", "ตู้", "บัส", "bus", "van"]),
    ("Car", ["เก๋ง", "ยนต์นั่ง", "sedan", "car", "suv"]),
    ("Bicycle / pedestrian", ["จักรยาน", "คนเดิน", "bicycle", "pedestrian"]),
]

# Bounding box around Thailand (with a small margin).
LAT_RANGE = (5.5, 20.6)
LON_RANGE = (97.3, 105.7)


@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_affected: int, note: str = "") -> None:
        self.steps.append({"step": step, "rows_affected": int(rows_affected), "note": note})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "rows_affected", "note"])


# ---------------------------------------------------------------- reading


def _norm_header(name: str) -> str:
    return re.sub(r"[\s_]", "", str(name).replace("﻿", "").strip().lower())


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read one CSV/XLS(X) file as strings, trying the encodings Thai agencies use."""
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path, dtype=str)
    for enc in ("utf-8-sig", "cp874", "tis-620"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("read_raw", b"", 0, 1, f"could not decode {path}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known headers to canonical names and drop everything else."""
    lookup = {_norm_header(a): canon for canon, aliases in COLUMN_ALIASES.items() for a in aliases}
    rename: dict[str, str] = {}
    for col in df.columns:
        canon = lookup.get(_norm_header(col))
        if canon and canon not in rename.values():
            rename[col] = canon
    out = df[list(rename)].rename(columns=rename)
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing or not ({"incident_datetime", "incident_date"} & set(out.columns)):
        raise ValueError(
            f"Could not find columns {missing or ['incident_date']} in file. "
            f"Headers present: {list(df.columns)}. Add the spelling to COLUMN_ALIASES."
        )
    return out


# ---------------------------------------------------------------- parsing


def _to_ascii(series: pd.Series) -> pd.Series:
    return series.astype("string").str.translate(THAI_DIGITS).str.strip()


def parse_date(series: pd.Series) -> pd.Series:
    """Parse the date formats found in Thai government files into datetimes.

    Handles ISO dates, d/m/Y with Buddhist-Era or Gregorian years, Thai month
    names ("5 ม.ค. 2566"), Thai digits and Excel serial numbers.
    """
    s = _to_ascii(series)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    serial = s.str.fullmatch(r"\d{5}(\.\d+)?").fillna(False)
    if serial.any():
        out[serial] = pd.to_datetime(s[serial].astype(float), unit="D", origin="1899-12-30")

    # "5 ม.ค. 2566" / "5 มกราคม 2566" -> "5/1/2566"
    for name, num in sorted(THAI_MONTHS.items(), key=lambda kv: -len(kv[0])):
        s = s.str.replace(rf"\s*{re.escape(name)}\s*", f"/{num}/", regex=True)

    # Buddhist-Era years (2400+) -> Gregorian.
    s = s.str.replace(r"\b(2[4-6]\d{2})\b", lambda m: str(int(m.group(1)) - 543), regex=True)

    rest = ~serial & s.notna()
    iso = rest & s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").fillna(False)
    dmy = rest & ~iso & s.str.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}").fillna(False)
    other = rest & ~iso & ~dmy

    out[iso] = pd.to_datetime(s[iso], errors="coerce", format="ISO8601")
    day_part = s[dmy].str.extract(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})")
    out[dmy] = pd.to_datetime(
        day_part[0] + "/" + day_part[1] + "/" + day_part[2], errors="coerce", format="%d/%m/%Y"
    )
    if other.any():
        out[other] = pd.to_datetime(s[other], errors="coerce", format="mixed", dayfirst=True)
    return out


def parse_time(series: pd.Series) -> pd.Series:
    """Parse "14:30", "14.30 น.", "1430", "14:30:00" into a Timedelta since midnight."""
    s = _to_ascii(series).str.replace(r"\s*(น\.|น|hrs?\.?)$", "", regex=True)
    parts = s.str.extract(r"^(\d{1,2})[:.]?(\d{2})?(?::(\d{2}))?$")
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    valid = hour.between(0, 23) & minute.between(0, 59)
    td = pd.to_timedelta(hour * 60 + minute, unit="min")
    return td.where(valid)


def _categorize(series: pd.Series, rules: list[tuple[str, list[str]]], unknown: str = "Unknown") -> pd.Series:
    def one(text: object) -> str:
        if pd.isna(text) or not str(text).strip() or str(text).strip() in {"-", "ไม่ทราบ", "nan"}:
            return unknown
        t = str(text).lower()
        for label, words in rules:
            if any(w in t for w in words):
                return label
        return "Other"

    uniques = series.dropna().unique()
    mapping = {u: one(u) for u in uniques}
    return series.map(mapping).fillna(unknown)


def _to_count(series: pd.Series) -> pd.Series:
    s = _to_ascii(series).str.replace(",", "", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    n = pd.to_numeric(s, errors="coerce")
    return n.where(n >= 0)


# ---------------------------------------------------------------- pipeline


def clean(raw: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """Clean one standardised raw table. See module docstring for the rules."""
    log = log or CleaningLog()
    df = standardize_columns(raw)
    log.add("rows read", len(df))

    # Strip whitespace everywhere and turn empty-ish markers into NaN.
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "-": pd.NA, "nan": pd.NA})

    # --- date & time
    if "incident_datetime" in df.columns:
        dt = parse_date(df["incident_datetime"])
        time_part = df["incident_datetime"].str.extract(r"(\d{1,2}[:.]\d{2}(?::\d{2})?)")[0]
        td = parse_time(time_part)
    else:
        dt = parse_date(df["incident_date"])
        td = parse_time(df["incident_time"]) if "incident_time" in df.columns else pd.Series(pd.NaT, index=df.index)
    bad_date = dt.isna() & df.get("incident_datetime", df.get("incident_date")).notna()
    log.add("unparseable dates -> dropped", bad_date.sum())
    log.add("missing or invalid time -> hour left blank", td.isna().sum())
    df["date"] = dt.dt.normalize()
    df["datetime"] = df["date"] + td
    df["hour"] = (td.dt.total_seconds() // 3600).astype("Int64")

    # --- province
    canon = normalize_province(df["province"])
    unmatched = canon.isna() & df["province"].notna()
    if unmatched.any():
        examples = ", ".join(df.loc[unmatched, "province"].value_counts().head(5).index)
        log.add("province names not recognised -> blank", unmatched.sum(), f"e.g. {examples}")
    log.add(
        "province names rewritten to canonical form",
        (canon.notna() & (canon != df["province"])).sum(),
        "prefixes like 'จ.', English names, nicknames such as กทม/โคราช",
    )
    df["province_raw"] = df["province"]
    df = df.drop(columns="province").join(canon.rename("province_th"))
    df = df.merge(PROVINCE_TABLE, on="province_th", how="left")

    # --- counts
    for col in ["fatalities", "injuries", "injuries_serious", "injuries_minor", "vehicles_involved"]:
        if col in df.columns:
            df[col] = _to_count(df[col])
    if "injuries" not in df.columns and {"injuries_serious", "injuries_minor"} <= set(df.columns):
        df["injuries"] = df["injuries_serious"].fillna(0) + df["injuries_minor"].fillna(0)
        log.add("injuries derived from serious + minor", len(df))
    for col in ["fatalities", "injuries"]:
        if col not in df.columns:
            df[col] = np.nan
        log.add(f"{col}: blank -> 0", df[col].isna().sum(), "a blank casualty cell means none were recorded")
        df[col] = df[col].fillna(0).astype(int)

    # --- coordinates
    if {"latitude", "longitude"} <= set(df.columns):
        lat = pd.to_numeric(_to_ascii(df["latitude"]), errors="coerce")
        lon = pd.to_numeric(_to_ascii(df["longitude"]), errors="coerce")
        swapped = lat.between(*LON_RANGE) & lon.between(*LAT_RANGE)
        lat, lon = lat.where(~swapped, lon), lon.where(~swapped, lat)
        log.add("latitude/longitude swapped back", swapped.sum())
        outside = lat.notna() & lon.notna() & ~(lat.between(*LAT_RANGE) & lon.between(*LON_RANGE))
        log.add("coordinates outside Thailand -> blank", outside.sum())
        df["latitude"], df["longitude"] = lat.where(~outside), lon.where(~outside)
    else:
        df["latitude"] = df["longitude"] = np.nan

    # --- categories (keep the raw text next to the grouped one)
    df["cause_group"] = _categorize(df["cause"] if "cause" in df.columns else pd.Series(pd.NA, index=df.index), CAUSE_RULES)
    df["vehicle_group"] = _categorize(df["vehicle"] if "vehicle" in df.columns else pd.Series(pd.NA, index=df.index), VEHICLE_RULES)

    # --- drop unusable rows, then duplicates
    df = df[df["date"].notna()]
    before = len(df)
    key = ["acc_code"] if "acc_code" in df.columns and df["acc_code"].notna().mean() > 0.9 else None
    df = df.drop_duplicates(subset=key)
    log.add("duplicate rows removed", before - len(df), f"key: {key or 'all columns'}")
    log.add("rows kept", len(df))
    return df.reset_index(drop=True)


def clean_files(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean several yearly files and stack them. Returns (data, cleaning log)."""
    frames, logs = [], []
    for path in paths:
        log = CleaningLog()
        frames.append(clean(read_raw(path), log))
        logs.append(log.to_frame().assign(file=Path(path).name))
    data = pd.concat(frames, ignore_index=True)
    before = len(data)
    if "acc_code" in data.columns and data["acc_code"].notna().mean() > 0.9:
        data = data.drop_duplicates(subset="acc_code").reset_index(drop=True)
    cross = pd.DataFrame(
        [{"step": "duplicates across files removed", "rows_affected": before - len(data), "note": "", "file": "(all)"}]
    )
    return data, pd.concat(logs + [cross], ignore_index=True)
