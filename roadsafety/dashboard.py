"""Build the interactive dashboard: docs/index.html (served by GitHub Pages).

    python -m roadsafety.dashboard              # reads every file in data/raw/

The page is one self-contained HTML file. The cleaned accidents are rolled up
into small aggregate tables, embedded as JSON, and filtered in the browser, so
the page needs no server and no data files next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .clean import clean_files
from .features import PERIODS, add_features, calendar_days
from .provinces import PROVINCE_TABLE

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TEMPLATE = Path(__file__).resolve().parent / "templates" / "dashboard.html"
OUT = ROOT / "docs" / "index.html"

REGIONS = ["North", "Northeast", "Central", "East", "West", "South"]


def _codes(series: pd.Series, labels: list) -> pd.Series:
    return series.map({v: i for i, v in enumerate(labels)}).fillna(-1).astype(int)


def build_payload(df: pd.DataFrame) -> dict:
    years = sorted(df["year"].unique().tolist())
    provinces = PROVINCE_TABLE.to_dict("records")
    vehicles = df["vehicle_group"].value_counts().index.tolist()
    causes = df["cause_group"].value_counts().index.tolist()

    d = pd.DataFrame({
        "y": _codes(df["year"], years),
        "p": _codes(df["period"], PERIODS),
        "v": _codes(df["province_th"], PROVINCE_TABLE["province_th"].tolist()),
        "h": df["hour"].fillna(-1).astype(int),
        "veh": _codes(df["vehicle_group"], vehicles),
        "c": _codes(df["cause_group"], causes),
        "deaths": df["fatalities"].astype(int),
    })

    def cube(keys: list[str]) -> list[list[int]]:
        g = d.groupby(keys).agg(n=("deaths", "size"), deaths=("deaths", "sum")).reset_index()
        return g.astype(int).to_numpy().tolist()

    days = calendar_days(df)
    days["y"] = _codes(days["date"].dt.year, years)
    days["p"] = _codes(days["period"], PERIODS)
    day_count = days.groupby(["y", "p"]).size().reset_index().astype(int).to_numpy().tolist()

    # Daily national series, split by region so the timeline can follow the region filter.
    start = df["date"].min()
    region = _codes(df["region"], REGIONS)
    daily = (
        pd.DataFrame({"t": (df["date"] - start).dt.days, "r": region, "deaths": df["fatalities"]})
        .groupby(["t", "r"]).agg(n=("deaths", "size"), deaths=("deaths", "sum")).reset_index()
    )
    campaign = days.loc[days["period"] != "Normal days"]
    campaign_days = [[int((t - start).days), int(PERIODS.index(p))] for t, p in zip(campaign["date"], campaign["period"])]

    fatal = df[df["is_fatal"] & df["latitude"].notna()]
    points = pd.DataFrame({
        "lat": (fatal["latitude"] * 1000).round().astype(int),
        "lon": (fatal["longitude"] * 1000).round().astype(int),
        "y": _codes(fatal["year"], years),
        "p": _codes(fatal["period"], PERIODS),
        "v": _codes(fatal["province_th"], PROVINCE_TABLE["province_th"].tolist()),
        "deaths": fatal["fatalities"].astype(int),
    })

    return {
        "start": start.strftime("%Y-%m-%d"),
        "end": df["date"].max().strftime("%Y-%m-%d"),
        "years": years,
        "periods": PERIODS,
        "regions": REGIONS,
        "provinces": [[p["province_th"], p["province_en"], REGIONS.index(p["region"])] for p in provinces],
        "vehicles": vehicles,
        "causes": causes,
        # rows: [year, period, province, hour, accidents, deaths]
        "hourCube": cube(["y", "p", "v", "h"]),
        # rows: [year, period, province, vehicle, cause, accidents, deaths]
        "catCube": cube(["y", "p", "v", "veh", "c"]),
        # rows: [year, period, calendar days]
        "dayCount": day_count,
        # rows: [day offset from start, region, accidents, deaths]
        "daily": daily.astype(int).to_numpy().tolist(),
        "campaignDays": campaign_days,
        # rows: [lat*1000, lon*1000, year, period, province, deaths]
        "points": points.to_numpy().tolist(),
    }


def build(paths: list[Path], out: Path = OUT) -> Path:
    data, _ = clean_files(paths)
    df = add_features(data)
    payload = json.dumps(build_payload(df), ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload.replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args()
    paths = args.files or sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".csv", ".xls", ".xlsx"})
    if not paths:
        raise SystemExit(f"No data files in {RAW_DIR}. Run `python -m roadsafety.download` first.")
    out = build(paths)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
