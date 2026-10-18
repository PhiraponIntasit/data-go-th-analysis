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
    by_year = analysis.festival_by_year(df)
    profile = analysis.festival_day_profile(df)
    hours = analysis.hour_profile(df)
    prov = analysis.province_table(df)
    causes = analysis.category_comparison(df, "cause_group")
    vehicles = analysis.category_comparison(df, "vehicle_group")
    regions = analysis.region_table(df)

    figs = [
        charts.daily_rates(rates, fig_dir),
        charts.festival_days(profile, rates.loc["Normal days", "deaths_per_day"], fig_dir) if len(profile) else None,
        charts.hour_profile(hours, fig_dir),
        charts.weekday_hour_heatmap(analysis.weekday_hour(df), fig_dir),
        charts.top_provinces(prov, fig_dir),
        charts.province_lift(prov, fig_dir),
        charts.dumbbell(causes, "Presumed cause: festival vs normal days", fig_dir, "07_causes.png"),
        charts.dumbbell(vehicles, "Vehicle involved: festival vs normal days", fig_dir, "08_vehicles.png"),
        charts.yearly(by_year, fig_dir) if len(by_year) else None,
        charts.fatal_map(df, fig_dir) if df["latitude"].notna().any() else None,
    ]
    fig = {p.stem: f"figures/{p.name}" for p in figs if p is not None}

    # ---- numbers used in the narrative
    fest_rates = rates.drop(index="Normal days", errors="ignore")
    worst_fest = fest_rates["death_lift"].idxmax() if len(fest_rates) else None
    peak_start, peak_share = _peak_window(hours, "Festival") if (hours["group"] == "Festival").any() else (None, 0)
    sev_hours = hours[(hours["group"] == "Festival") & (hours["accidents"] >= 30)]
    deadliest_hour = sev_hours.loc[sev_hours["deaths_per_100_accidents"].idxmax()] if len(sev_hours) else None
    top5_burden = prov.nlargest(5, "festival_deaths")
    top5_lift = prov.dropna(subset=["lift"]).nlargest(5, "lift")
    known_causes = causes.drop(index=["Unknown", "Other"], errors="ignore")
    top_cause = known_causes["share_Festival"].idxmax() if len(known_causes) else None
    rising_cause = known_causes["share_change_pts"].idxmax() if len(known_causes) else None
    moto_share = vehicles["share_Festival"].get("Motorcycle", float("nan"))
    deadliest_day = profile.loc[profile["deaths_per_day"].idxmax()] if len(profile) else None
    no_hour = df["hour"].isna().mean()
    no_coords = df["latitude"].isna().mean()
    unknown_cause = (df["cause_group"] == "Unknown").mean()

    def names(t: pd.DataFrame) -> str:
        return ", ".join(f"{r.province_en} ({th})" for th, r in t.iterrows())

    L: list[str] = []
    add = L.append
    add("# Road accidents on Thai national highways during Songkran and New Year\n")
    add(
        f"Source: Department of Highways accident records published on data.go.th. "
        f"{ov['accidents']:,} accidents, {ov['deaths']:,} deaths and {ov['injuries']:,} injuries "
        f"from {ov['first_date']} to {ov['last_date']} in {ov['provinces']} provinces.\n"
    )
    add("> Generated by `python -m roadsafety.report`. Every figure below is computed from the data.\n")

    add("## Key findings\n")
    if worst_fest is not None:
        for p, r in fest_rates.iterrows():
            add(
                f"- **{p}**: {r['accidents_per_day']:.1f} accidents and {r['deaths_per_day']:.1f} deaths per day, "
                f"{r['accident_lift']:.2f}× the accident rate and {r['death_lift']:.2f}× the death rate of a normal day."
            )
    if deadliest_day is not None:
        add(
            f"- The deadliest day of the campaign window is **day {int(deadliest_day['festival_day'])} of "
            f"{deadliest_day['period']}** with {deadliest_day['deaths_per_day']:.1f} deaths on average."
        )
    if peak_start is not None:
        add(
            f"- During festivals {100 * peak_share:.0f}% of accidents happen in just three hours, "
            f"**{peak_start:02d}:00–{(peak_start + 3) % 24:02d}:00**."
        )
    if deadliest_hour is not None:
        add(
            f"- Accidents are most lethal at **{int(deadliest_hour['hour']):02d}:00** "
            f"({deadliest_hour['deaths_per_100_accidents']:.1f} deaths per 100 accidents)."
        )
    add(f"- Most festival deaths: {names(top5_burden)}.")
    if len(top5_lift):
        add(f"- Biggest festival surge relative to their own normal days: {names(top5_lift)}.")
    if top_cause:
        add(
            f"- Most common presumed cause during festivals: **{top_cause}** "
            f"({100 * causes.loc[top_cause, 'share_Festival']:.0f}% of accidents). "
            f"The cause that grows most during festivals is **{rising_cause}** "
            f"({causes.loc[rising_cause, 'share_change_pts']:+.1f} percentage points)."
        )
    if pd.notna(moto_share):
        add(f"- Motorcycles are the first vehicle recorded in {100 * moto_share:.0f}% of festival accidents.")
    add("")

    add("## 1. How much worse are festival days?\n")
    add(f"![daily rates]({fig['01_daily_rates']})\n")
    add(md_table(
        rates[["days", "accidents", "deaths", "accidents_per_day", "deaths_per_day", "deaths_per_100_accidents", "accident_lift", "death_lift"]],
        {"accidents": "{:,.0f}", "deaths": "{:,.0f}", "days": "{:,.0f}", "accidents_per_day": "{:.1f}",
         "deaths_per_day": "{:.1f}", "deaths_per_100_accidents": "{:.1f}", "accident_lift": "{:.2f}×", "death_lift": "{:.2f}×"},
    ) + "\n")
    if "09_festival_deaths_by_year" in fig:
        add(f"![by year]({fig['09_festival_deaths_by_year']})\n")
    if "02_festival_days" in fig:
        add(f"![festival days]({fig['02_festival_days']})\n")

    add("## 2. When: hour and weekday\n")
    add(f"![hours]({fig['03_hour_profile']})\n")
    add(f"![heatmap]({fig['04_weekday_hour_heatmap']})\n")

    add("## 3. Where: provinces and regions\n")
    add("Two different questions, two different rankings. *Burden* is where most people die. *Lift* compares a "
        "province with its own normal days, so it shows where the festival itself changes the risk, without needing "
        "population or road length.\n")
    add(f"![top provinces]({fig['05_top_provinces']})\n")
    add(f"![lift]({fig['06_province_lift']})\n")
    if "10_fatal_map" in fig:
        add(f"![map]({fig['10_fatal_map']})\n")
    add(md_table(regions, {"accidents": "{:,.0f}", "deaths": "{:,.0f}", "injuries": "{:,.0f}",
                           "fatal_accidents": "{:,.0f}", "deaths_per_100_accidents": "{:.1f}"}) + "\n")

    add("## 4. Why: causes and vehicles\n")
    add(f"![causes]({fig['07_causes']})\n")
    pct = "{:.1%}"
    add(md_table(causes[["share_Festival", "share_Normal days", "share_change_pts", "deaths_per_100_accidents"]],
                 {"share_Festival": pct, "share_Normal days": pct, "share_change_pts": "{:+.1f}", "deaths_per_100_accidents": "{:.1f}"}) + "\n")
    add(f"![vehicles]({fig['08_vehicles']})\n")

    add("## 5. Policy recommendations\n")
    recs = []
    if peak_start is not None:
        recs.append(
            f"**Put checkpoints where the hours are.** Schedule festival checkpoint and patrol shifts to cover "
            f"{peak_start:02d}:00–{(peak_start + 3) % 24:02d}:00, which alone holds {100 * peak_share:.0f}% of festival accidents"
            + (f", and keep night patrols running through {int(deadliest_hour['hour']):02d}:00 when crashes are most lethal."
               if deadliest_hour is not None else ".")
        )
    rec = (
        f"**Two budgets for two kinds of province.** High-burden provinces ({', '.join(top5_burden['province_en'])}) "
        f"need year-round engineering fixes on their highway black spots"
    )
    if len(top5_lift):
        rec += (
            f"; high-lift provinces ({', '.join(top5_lift['province_en'])}) need a temporary festival surge "
            f"of checkpoints, rest stops and emergency medical teams"
        )
    recs.append(rec + ".")
    if top_cause:
        rec = (
            f"**Target the behaviour that grows.** {rising_cause} gains the most share during festivals; "
            f"campaign messaging and enforcement should be aimed at it"
        )
        recs.append(rec + (f", alongside {top_cause} as the largest cause overall." if top_cause != rising_cause else "."))
    if pd.notna(moto_share):
        recs.append(
            f"**Motorcycles first.** With motorcycles in {100 * moto_share:.0f}% of festival accidents, helmet "
            f"enforcement and safe-return campaigns for riders give the largest reach per baht."
        )
    if deadliest_day is not None:
        recs.append(
            f"**Time the messages.** Day {int(deadliest_day['festival_day'])} of {deadliest_day['period']} is the deadliest; "
            f"push warnings and extra patrols the day before it, not only at the start of the holiday."
        )
    recs.append(
        f"**Fix the data at the source.** {100 * no_hour:.0f}% of records have no usable time, "
        f"{100 * no_coords:.0f}% no usable coordinates and {100 * unknown_cause:.0f}% no presumed cause. "
        f"Publishing one schema across years with ISO dates and validated GPS would make analyses like this one cheaper and more reliable."
    )
    for i, r in enumerate(recs, 1):
        add(f"{i}. {r}")
    add("")

    add("## 6. Data cleaning log\n")
    add(md_table(log, {"rows_affected": "{:,}"}, index=False) + "\n")
    add("## Limitations\n")
    add("- Covers only accidents recorded by the Department of Highways on its own network; rural roads (DRR), "
        "municipal roads and expressways are not included, so these are not national totals.")
    add("- Deaths are those recorded at the scene/report time; the 30-day death count used by WHO is higher.")
    add("- Presumed cause is the officer's judgement at the scene and groups were assigned by keyword rules "
        "(`roadsafety/clean.py`), so treat cause shares as indicative.")
    add("- Province rankings by count reflect highway length and traffic as well as risk; the lift metric is the fairer comparison.")

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
