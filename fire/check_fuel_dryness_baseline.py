#!/usr/bin/env python3
"""Trend-controlled check: is fuel moisture on a fire's ignition date drier than a *typical*
year at the same cell and calendar date, not just drier than the multi-decade average?

Later years in the moisture cache are systematically drier (see CLAUDE.md, "37-year run"),
so comparing a fire against the full cache's climatology risks mistaking the long-term
drying trend for a fire-specific signal. This instead computes, for each event, the same
dryness-rank statistic (plot_fire_event_maps.py's "share of other years that were wetter")
using only a fixed recent baseline window (default 2000-2024) as the "other years" set, then
compares the event's own score against the mean score a plain calendar date in that window
gets. If fires still score above that baseline on average, the dryness signal on fire days is
not simply an artefact of the warming trend.

Usage
-----
    python3 check_fuel_dryness_baseline.py --catalog fire/data/catalog.json \\
        --stats fire/work/stats.json --cache fire/work/fire_daily_moisture.npz \\
        --baseline-start-year 2000 --baseline-end-year 2024 --out fire/work/dryness_baseline_check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_dryness_common as fdc
from plot_fire_event_maps import EPS


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=Path("fire/data/catalog.json"))
    p.add_argument("--stats", type=Path, required=True, help="output of plot_fire_event_maps.py --stats-out "
                    "(used only for each event's already-resolved grid cell)")
    p.add_argument("--cache", type=Path, required=True, help="output of extract_fire_daily_moisture.py")
    p.add_argument("--baseline-start-year", type=int, default=2000)
    p.add_argument("--baseline-end-year", type=int, default=2024)
    p.add_argument("--out", type=Path, default=Path("fire/work/dryness_baseline_check.json"))
    return p.parse_args()


def dryness_share(values_by_year: np.ndarray, event_year_index: int, eps: float) -> float:
    """`values_by_year` is indexed by consecutive years starting at the cache's first year;
    NaN where that year has no data for this calendar date."""
    v = values_by_year[event_year_index]
    others = np.delete(values_by_year, event_year_index)
    others = others[np.isfinite(others)]
    return ((others > v + eps).sum() + 0.5 * (np.abs(others - v) <= eps).sum()) / len(others)


def main() -> None:
    args = get_args()
    catalog = json.load(args.catalog.open())
    stats = json.load(args.stats.open())
    z = np.load(args.cache)
    dates = z["dates"].astype("datetime64[D]")
    lat, lon = z["lat"], z["lon"]
    first_year = int(dates[0].astype("datetime64[Y]").astype(int)) + 1970

    results = {}
    for var in ("DFMC_10_min", "DFMC_1000_mean", "LFMC_L_mean"):
        var_cache = np.asarray(z[var])  # decompress once; indexing the lazy npz array per-lookup is far slower
        records = []
        for e in catalog:
            s = stats.get(e["key"])
            if s is None or "error" in s:
                continue
            j = int(np.abs(lat - s["cell_lat"]).argmin())
            i = int(np.abs(lon - s["cell_lon"]).argmin())
            month_day = e["date"][5:]
            event_year = int(e["date"][:4])
            last_year = event_year  # extend the per-year lookup range to at least the event's own year
            years = range(first_year, max(args.baseline_end_year, last_year) + 1)
            values = np.array([
                np.nan if (t := fdc.day_index(dates, y, month_day)) is None else var_cache[t, j, i]
                for y in years
            ], float)
            event_idx = event_year - first_year
            if not np.isfinite(values[event_idx]):
                continue
            event_share = dryness_share(values, event_idx, EPS[var])
            baseline_shares = [
                dryness_share(values, y - first_year, EPS[var])
                for y in range(args.baseline_start_year, args.baseline_end_year + 1)
                if np.isfinite(values[y - first_year])
            ]
            records.append((event_year, event_share, float(np.mean(baseline_shares))))

        arr = np.array(records)
        event_year_col, event_share_col, baseline_col = arr[:, 0], arr[:, 1], arr[:, 2]
        fire_years = np.unique(event_year_col)
        excess_by_year = np.array([(event_share_col[event_year_col == y] - baseline_col[event_year_col == y]).mean() for y in fire_years])
        results[var] = dict(
            n_events=len(records), mean_event_share=float(event_share_col.mean()), mean_baseline_share=float(baseline_col.mean()),
            mean_excess=float((event_share_col - baseline_col).mean()), frac_events_above_baseline=float((event_share_col > baseline_col).mean()),
            n_fire_years=len(fire_years), n_fire_years_above_baseline=int((excess_by_year > 0).sum()),
            median_year_excess=float(np.median(excess_by_year)),
        )
        r = results[var]
        print(f"{var:16s} events {r['mean_event_share']:.2f} vs baseline ({args.baseline_start_year}-{args.baseline_end_year}) "
              f"{r['mean_baseline_share']:.2f} -> excess {r['mean_excess']:+.2f}; above baseline in "
              f"{r['frac_events_above_baseline']:.0%} of events; by fire-year: {r['n_fire_years_above_baseline']}/{r['n_fire_years']} "
              f"positive, median excess {r['median_year_excess']:+.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, args.out.open("w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
