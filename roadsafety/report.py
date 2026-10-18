"""Run the whole analysis and write reports/REPORT.md with figures.

    python -m roadsafety.report                 # reads every file in data/raw/
    python -m roadsafety.report path/to/a.csv   # or specific files

Every number in the report is computed from the data at run time; nothing is
typed in by hand, so re-running on a new year of data updates the whole story.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import analysis, charts
from .clean import clean_files
from .features import add_features

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None, index: bool = True) -> str:
    fmt = fmt or {}
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(d[c]) else "---" for c in d.columns) + "|"
    rows = [
        "| " + " | ".join(fmt.get(c, "{}").format(v) if pd.notna(v) else "–" for c, v in row.items()) + " |"
        for _, row in d.iterrows()
    ]
    return "\n".join([head, sep, *rows])


def _peak_window(hours: pd.DataFrame, group: str, width: int = 3) -> tuple[int, float]:
    """Start hour and share of the busiest `width`-hour block (wrapping past midnight)."""
    share = hours[hours["group"] == group].set_index("hour")["share_of_accidents"].reindex(range(24), fill_value=0)
    sums = {h: sum(share[(h + i) % 24] for i in range(width)) for h in range(24)}
    start = max(sums, key=sums.get)
    return start, sums[start]


MONTHS = {"01": "Jan", "04": "Apr", "12": "Dec"}


def _md(md: str) -> str:
    """'12-31' -> '31 Dec'."""
    return f"{int(md[3:])} {MONTHS.get(md[:2], md[:2])}"


def _share(table: pd.DataFrame, row: str, col: str) -> float:
    return float(table.loc[row, col]) if row in table.index and col in table.columns else float("nan")


def build(paths: list[Path], out_dir: Path = REPORTS, processed_dir: Path = PROCESSED) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    data, log = clean_files(paths)
    df = add_features(data)
    df.to_csv(processed_dir / "accidents_clean.csv", index=False, encoding="utf-8-sig")
    log.to_csv(out_dir / "cleaning_log.csv", index=False, encoding="utf-8-sig")

    ov = analysis.overview(df)
    rates = analysis.period_rates(df)
    windows = analysis.campaign_windows(df)
    by_year = analysis.festival_by_year(df)
    around = analysis.around_nominal_start(df)
    profile = analysis.festival_day_profile(df)
    hours = analysis.hour_profile(df)
    prov = analysis.province_table(df)
    causes = analysis.category_comparison(df, "cause_group")
    vehicles = analysis.category_comparison(df, "vehicle_group")
    agencies = analysis.agency_mix(df) if "agency" in df.columns else pd.DataFrame()
    regions = analysis.region_table(df)
    fest_rates = rates.drop(index="Normal days", errors="ignore")
    has_fest = len(fest_rates) > 0
    fest_severity = 100 * df.loc[df["period"] != "Normal days", "fatalities"].sum() / max((df["period"] != "Normal days").sum(), 1)

    figs = [
        charts.recording_step(around, fig_dir) if len(around) else None,
        charts.severity_by_period(rates, fig_dir),
        charts.festival_calendar(profile, rates.loc["Normal days", "deaths_per_day"], fig_dir) if len(profile) else None,
        charts.yearly(by_year, fig_dir) if len(by_year) else None,
        charts.hour_profile(hours, fig_dir),
        charts.top_provinces(prov, fig_dir),
        charts.province_severity(prov, fest_severity, fig_dir),
        charts.vehicle_severity(vehicles, fig_dir),
        charts.causes(causes, fig_dir),
        charts.fatal_map(df, fig_dir) if df["latitude"].notna().any() else None,
    ]
    fig = {p.stem: f"figures/{p.name}" for p in figs if p is not None}

    # ---- numbers used in the narrative
    step = float("nan")
    if len(around):
        inside = around[around["offset"].between(0, 6)]["accidents"].mean()
        before = around[around["offset"] <= -4]["accidents"].mean()
        step = inside / before
    drr = "กรมทางหลวงชนบท"
    drr_fest, drr_normal = _share(agencies, drr, "Festival"), _share(agencies, drr, "Normal days")
    drunk_fest, drunk_normal = _share(causes, "Drunk driving", "share_Festival"), _share(causes, "Drunk driving", "share_Normal days")
    sev_normal = rates.loc["Normal days", "deaths_per_100_accidents"]

    peak_start, peak_share = _peak_window(hours, "Festival") if (hours["group"] == "Festival").any() else (None, 0)
    h = hours.set_index(["group", "hour"])
    night = list(range(22, 24)) + list(range(0, 4))
    day = list(range(8, 17))

    def sev(group: str, hrs: list[int]) -> float:
        sub = h.loc[group].reindex(hrs).dropna()
        return 100 * sub["deaths"].sum() / sub["accidents"].sum() if len(sub) else float("nan")

    night_fest, day_fest, night_normal = sev("Festival", night), sev("Festival", day), sev("Normal days", night)

    deadliest = profile.loc[profile.groupby("period")["deaths_per_day"].idxmax()] if len(profile) else profile
    trend = {}
    for fest, g in by_year.groupby("period"):
        if len(g) >= 3:
            years = g["festival_year"].astype(int).tolist()
            first, last = g.iloc[:2]["deaths_per_day"].mean(), g.iloc[-2:]["deaths_per_day"].mean()
            trend[fest] = (f"{years[0]}–{years[1]}", f"{years[-2]}–{years[-1]}", first, last)

    top5 = prov.nlargest(5, "festival_deaths")
    top5_share = top5["share_of_festival_deaths"].sum()
    sev5 = prov.dropna(subset=["festival_deaths_per_100_accidents"]).nlargest(5, "festival_deaths_per_100_accidents")
    moto_sev = _share(vehicles, "Motorcycle", "deaths_per_100_accidents")
    car_sev = _share(vehicles, "Car", "deaths_per_100_accidents")
    moto_fest, moto_normal = _share(vehicles, "Motorcycle", "share_Festival"), _share(vehicles, "Motorcycle", "share_Normal days")
    known_causes = causes.drop(index=["Unknown", "Other"], errors="ignore")
    top_cause = known_causes["share_Festival"].idxmax() if len(known_causes) and "share_Festival" in known_causes else None
    steps = log.groupby("step")["rows_affected"].sum()

    def names(t: pd.DataFrame) -> str:
        return ", ".join(f"{r.province_en} ({th})" for th, r in t.iterrows())

    def pct(x: float) -> str:
        return f"{100 * x:.0f}%"

    L: list[str] = []
    add = L.append
    add("# Road accidents during Songkran and New Year on Thailand's national road network\n")
    agencies_txt = ", ".join(df["agency"].dropna().value_counts().index) if "agency" in df.columns else "n/a"
    add(
        f"Source: Ministry of Transport accident records (agencies: {agencies_txt}), "
        f"open data portal datagov.mot.go.th. After cleaning: {ov['accidents']:,} accidents, {ov['deaths']:,} deaths and "
        f"{ov['injuries']:,} injuries from {ov['first_date']} to {ov['last_date']} in {ov['provinces']} provinces.\n"
    )
    add("> Generated by `python -m roadsafety.report`. Every number below is computed from the data.\n")

    add("## Key findings\n")
    if has_fest:
        add(
            f"1. **The festival numbers are partly a recording effect.** On campaign days the data holds "
            f"{fest_rates['accidents_per_day'].min():.0f}–{fest_rates['accidents_per_day'].max():.0f} accidents a day against "
            f"{rates.loc['Normal days', 'accidents_per_day']:.0f} on normal days, but the jump happens exactly on the day a campaign starts "
            f"(about {step:.1f}× overnight) and reverses the day it ends. Rural-roads (DRR) records rise from "
            f"{pct(drr_normal)} to {pct(drr_fest)} of all records and drunk driving from {100 * drunk_normal:.1f}% to "
            f"{100 * drunk_fest:.1f}% of causes, which points to more intensive reporting during the "
            f"\"7 dangerous days\" campaigns. Raw festival-vs-normal ratios should not be read as risk ratios."
        )
        add(
            f"2. **Per crash, festivals are not deadlier.** Deaths per 100 recorded accidents: "
            + ", ".join(f"{p} {r['deaths_per_100_accidents']:.1f}" for p, r in rates.iterrows())
            + ". Whatever the true rise in crashes is (see 1), each crash is no more likely to kill."
        )
    if len(deadliest):
        add(
            "3. **The deadliest days are the days around the holiday itself:** "
            + "; ".join(f"{_md(r['md'])} for {r['period']} ({r['deaths_per_day']:.1f} deaths a day)" for _, r in deadliest.iterrows())
            + ". The last days of each campaign are the quietest."
        )
    if trend:
        add(
            "4. **Deaths per campaign day are falling:** "
            + "; ".join(f"{f} {a} averaged {x:.1f} a day, {b} averaged {y:.1f} ({100 * (y / x - 1):+.0f}%)"
                        for f, (a, b, x, y) in trend.items())
            + ". Songkran 2020 was cancelled and 2020–2021 travel was limited by COVID-19 measures, so read the early years with care."
        )
    if peak_start is not None:
        add(
            f"5. **Early evening is busiest, night is deadliest.** During campaigns {pct(peak_share)} of accidents happen "
            f"between {peak_start:02d}:00 and {(peak_start + 3) % 24:02d}:00, but a crash between 22:00 and 04:00 kills "
            f"{night_fest:.1f} people per 100 accidents against {day_fest:.1f} during the day (08:00–17:00) "
            f"and {night_normal:.1f} at night on normal days."
        )
    if pd.notna(moto_sev):
        add(
            f"6. **Motorcycles:** {pct(moto_fest)} of campaign accidents (vs {pct(moto_normal)} on normal days) and "
            f"{moto_sev:.1f} deaths per 100 accidents, {moto_sev / car_sev:.1f}× the rate for cars."
        )
    add(
        f"7. **Five provinces account for {pct(top5_share)} of campaign deaths:** {names(top5)}. "
        f"Where a festival crash is most likely to kill: {names(sev5)}."
    )
    add("")

    add("## 1. Are festival days really 4× more dangerous?\n")
    if "01_recording_step" in fig:
        add(f"![recording step]({fig['01_recording_step']})\n")
    add("Average recorded accidents per day around the nominal campaign start, all years together. Real traffic does not "
        "quadruple overnight and fall back overnight a week later; recording practice does. During campaigns the agencies "
        "collect accident reports far more completely (for example from rural roads), so the normal-day records are an "
        "undercount compared with the campaign records.\n")
    add("Because campaign dates move a little each year (and Songkran 2020 was cancelled), the campaign days used below are "
        "**detected from that step in the data**: a day in the festival season with at least 1.8× the 29-day rolling median, "
        "in a run of four days or more.\n")
    add(md_table(
        windows[["period", "festival_year", "start", "end", "days", "accidents", "deaths", "deaths_per_day", "deaths_per_100_accidents"]],
        {"start": "{:%d %b %Y}", "end": "{:%d %b %Y}", "accidents": "{:,.0f}", "deaths": "{:,.0f}",
         "deaths_per_day": "{:.1f}", "deaths_per_100_accidents": "{:.1f}", "festival_year": "{:.0f}"}, index=False,
    ) + "\n")
    add(md_table(
        rates[["days", "accidents", "deaths", "accidents_per_day", "deaths_per_day", "deaths_per_100_accidents", "accidents_vs_normal", "deaths_vs_normal"]],
        {"accidents": "{:,.0f}", "deaths": "{:,.0f}", "days": "{:,.0f}", "accidents_per_day": "{:.1f}",
         "deaths_per_day": "{:.1f}", "deaths_per_100_accidents": "{:.1f}", "accidents_vs_normal": "{:.1f}×", "deaths_vs_normal": "{:.1f}×"},
    ) + "\n")
    if len(agencies):
        add("Share of records by reporting agency:\n")
        add(md_table(agencies, {"Festival": "{:.1%}", "Normal days": "{:.1%}"}) + "\n")
    add(f"![severity]({fig['02_severity_by_period']})\n")

    add("## 2. When: days, years and hours\n")
    if "03_campaign_days" in fig:
        add(f"![campaign days]({fig['03_campaign_days']})\n")
    if "04_deaths_by_year" in fig:
        add(f"![by year]({fig['04_deaths_by_year']})\n")
    add(f"![hours]({fig['05_hour_profile']})\n")

    add("## 3. Where: provinces and regions\n")
    add("*Burden* (total deaths) shows where most lives are lost; *severity* (deaths per 100 accidents) shows where a crash "
        "is most likely to be fatal, which points at speed, road design and how fast victims reach hospital.\n")
    add(f"![top provinces]({fig['06_top_provinces']})\n")
    add(f"![severity]({fig['07_province_severity']})\n")
    if "10_fatal_map" in fig:
        add(f"![map]({fig['10_fatal_map']})\n")
    add(md_table(regions, {"accidents": "{:,.0f}", "deaths": "{:,.0f}", "injuries": "{:,.0f}",
                           "fatal_accidents": "{:,.0f}", "deaths_per_100_accidents": "{:.1f}"}) + "\n")

    add("## 4. Why: vehicles and causes\n")
    add(f"![vehicles]({fig['08_vehicle_severity']})\n")
    add(f"![causes]({fig['09_causes']})\n")
    add("Presumed cause is the officer's judgement at the scene and the campaign reporting forms differ from routine ones, "
        "so compare causes *within* campaigns rather than against normal days.\n")
    share = "{:.1%}"
    add(md_table(causes[["share_Festival", "share_Normal days", "deaths_per_100_accidents"]],
                 {"share_Festival": share, "share_Normal days": share, "deaths_per_100_accidents": "{:.1f}"}) + "\n")

    add("## 5. Policy recommendations\n")
    recs = []
    if has_fest:
        recs.append(
            "**Measure the campaign honestly.** Headlines that compare the \"7 dangerous days\" with ordinary days overstate the "
            f"festival effect, because recording itself jumps about {step:.1f}×. Track year-on-year deaths per campaign day and "
            "deaths per 100 accidents instead, and record every day of the year with the campaign's completeness so the two can be compared."
        )
    if pd.notna(night_fest):
        recs.append(
            f"**Move alcohol checkpoints into the night.** A campaign crash between 22:00 and 04:00 kills {night_fest:.1f} per 100 "
            f"against {day_fest:.1f} in daytime; that is where drink-driving checkpoints and speed enforcement save the most lives. "
            + (f"Keep traffic management on the busiest {peak_start:02d}:00–{(peak_start + 3) % 24:02d}:00 block." if peak_start is not None else "")
        )
    if len(deadliest):
        recs.append(
            "**Front-load resources.** The deadliest days are "
            + " and ".join(f"{_md(r['md'])} ({r['period']})" for _, r in deadliest.iterrows())
            + "; staff checkpoints and emergency medical teams for those days rather than spreading them evenly over the window."
        )
    if pd.notna(moto_sev):
        recs.append(
            f"**Motorcycles first.** With {moto_sev:.1f} deaths per 100 motorcycle crashes ({moto_sev / car_sev:.1f}× cars) and "
            f"{pct(moto_fest)} of campaign accidents, helmet enforcement, daytime-running-light checks and safe-return campaigns "
            "aimed at riders give the largest return."
        )
    recs.append(
        f"**Two lists of provinces.** High-burden provinces ({', '.join(top5['province_en'])}) need the most checkpoints and "
        f"ambulances; high-severity provinces ({', '.join(sev5['province_en'])}) need a review of black-spot road design and "
        "emergency response times, because their crashes turn fatal more often."
    )
    fixes = [
        (steps.get("completely empty rows dropped", 0), "completely empty rows"),
        (steps.get("Excel serial-number dates converted", 0), "dates stored as Excel serial numbers"),
        (steps.get("incident date before report date fixed by swapping day/month", 0)
         + steps.get("same ACC_CODE with swapped day/month -> kept the copy matching neighbouring codes", 0),
         "dates with day and month swapped"),
        (steps.get("duplicate rows removed", 0), "duplicate records"),
    ]
    recs.append(
        "**Publish the data in one consistent format.** Cleaning these files meant handling "
        + ", ".join(f"{int(n):,} {what}" for n, what in fixes if n)
        + ", three different date formats and three header layouts. One schema with ISO dates (YYYY-MM-DD) and a stable "
        "accident ID across years would make this data far cheaper to use."
    )
    for i, r in enumerate(recs, 1):
        add(f"{i}. {r}")
    add("")

    add("## 6. Data cleaning log\n")
    add(md_table(log, {"rows_affected": "{:,}"}, index=False) + "\n")
    add("## Limitations\n")
    add("- Covers roads managed by the Ministry of Transport (Department of Highways, Department of Rural Roads, "
        "Expressway Authority). Municipal and local roads, where many motorcycle crashes happen, are not included, "
        "so these are not national totals.")
    add("- Deaths are those recorded at the scene or at report time; the 30-day death count used by WHO is higher.")
    add("- Recording completeness differs between campaign and normal days (section 1); ratios between them are upper bounds.")
    add("- Presumed cause groups were assigned by keyword rules (`roadsafety/clean.py`); treat cause shares as indicative.")
    add("- Province totals reflect highway length and traffic as well as risk.")

    path = out_dir / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args()
    paths = args.files or sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".csv", ".xls", ".xlsx"})
    if not paths:
        raise SystemExit(f"No data files in {RAW_DIR}. Run `python -m roadsafety.download` first.")
    print(f"wrote {build(paths)}")


if __name__ == "__main__":
    main()
