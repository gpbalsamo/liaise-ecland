#!/usr/bin/env python3
"""Overview of every catalogued fire: a map and a timeline of fuel-moisture dryness rank,
plus a flat CSV of the per-event statistics.

Reads the catalogue (fire/build_fire_catalog.py) and the per-event stats JSON written by
fire/plot_fire_event_maps.py (its `--stats-out`); does not touch the daily-moisture cache
directly. "Dryness rank" here is the share of the other cached years that were wetter on the
fire's own ignition date (100 % = the driest that date has been across the whole cache).

Usage
-----
    python3 plot_fire_overview.py --catalog fire/data/catalog.json \\
        --stats fire/work/stats.json --out-dir fire/work
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_dryness_common as fdc

VARS = (("DFMC_10_min", "fast10h", "Fast dead fuel (10 h)"),
        ("DFMC_1000_mean", "slow1000h", "Slow dead fuel (1000 h)"),
        ("LFMC_L_mean", "live_low", "Live fuel moisture, low vegetation"))


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=Path("fire/data/catalog.json"))
    p.add_argument("--stats", type=Path, required=True, help="output of plot_fire_event_maps.py --stats-out")
    p.add_argument("--out-dir", type=Path, default=Path("fire/work"))
    return p.parse_args()


def marker_size(area_ha: np.ndarray) -> np.ndarray:
    return 18 * (np.asarray(area_ha) / 1000.0) ** 0.6


def main() -> None:
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog = json.load(args.catalog.open())
    stats = json.load(args.stats.open())

    rows = []
    for e in catalog:
        s = stats.get(e["key"])
        if s is None or "error" in s:
            continue
        row = dict(key=e["key"], date=e["date"], name=e["name"], origin=e["origin"], area_ha=e["area_ha"],
                   lon=e["lon"], lat=e["lat"], cell_lat=s["cell_lat"], cell_lon=s["cell_lon"],
                   cell_moved_to_land=s["moved_to_land"], n_other_years=s["n_other_years"])
        for var, label, _ in VARS:
            t = s["stats"][var]
            row.update({f"{label}_value": t["value"], f"{label}_otheryears_mean": t["other_years_mean"],
                        f"{label}_drier_than_n": t["drier_than_n"], f"{label}_tied": t["tied"],
                        f"{label}_dryness_share": t["dryness_share"]})
        rows.append(row)
    if not rows:
        sys.exit("no events with usable stats -- check --stats")

    csv_path = args.out_dir / "events_fuel_dryness.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    area = np.array([r["area_ha"] for r in rows], float)
    year = np.array([int(r["date"][:4]) for r in rows])
    date = np.array([np.datetime64(r["date"]) for r in rows])
    share = {label: np.array([r[f"{label}_dryness_share"] for r in rows]) for _, label, _ in VARS}

    print(f"events: {len(rows)} | moved to nearest land cell: {sum(r['cell_moved_to_land'] for r in rows)}")
    for _, label, _ in VARS:
        w = share[label]
        years = np.unique(year)
        year_mean = np.array([w[year == y].mean() for y in years])
        print(f"{label:10s} event-level: median {np.median(w):.2f} mean {w.mean():.2f} | "
              f">=0.5: {(w > 0.5).sum()}/{len(w)} | >=0.8: {(w >= 0.8).sum()} | >=0.95: {(w >= 0.95).sum()} | "
              f"year-level (n={len(years)}): median {np.median(year_mean):.2f}, years above 0.5: {(year_mean > 0.5).sum()}/{len(years)}")

    # ---- map: two panels, one per validation variable ------------------------------------------------------
    ink, ink2, ink3, surface, axis, grid, blue = fdc.INK, fdc.INK2, fdc.INK3, fdc.SURFACE, fdc.AXIS, fdc.GRID, fdc.BLUE
    lon0, lon1 = min(r["lon"] for r in rows) - 1, max(r["lon"] for r in rows) + 1
    lat0, lat1 = min(r["lat"] for r in rows) - 1, max(r["lat"] for r in rows) + 1
    extent = [lon0, lon1, lat0, lat1]
    order = np.argsort(-area)  # draw big markers first, small on top

    fig = plt.figure(figsize=(16, 7.4))
    fig.text(0.04, 0.965, f"{len(rows)} notable fires: how unusually dry was the fuel on the day?",
              fontsize=15, fontweight="semibold", va="top")
    fig.text(0.04, 0.925,
              "Colour: share of the other cached years in which that date was wetter at the fire’s grid cell "
              "(100 % = driest on record). Marker size grows with burned area.",
              fontsize=9.2, color=ink2, va="top")
    for q, (_, label, panel_title) in enumerate([v for v in VARS if v[1] != "fast10h"]):
        ax = fig.add_axes([0.05 + 0.475 * q, 0.20, 0.44, 0.65], projection=fdc.ccrs.PlateCarree())
        fdc.draw_basemap(ax, extent, with_cities=(q == 0))
        sc = ax.scatter(np.array([r["lon"] for r in rows])[order], np.array([r["lat"] for r in rows])[order],
                        s=marker_size(area[order]), c=share[label][order], cmap=fdc.DIVERGING_DRY_WET.reversed(),
                        norm=Normalize(0, 1), edgecolors=surface, linewidths=0.8, alpha=0.92,
                        transform=fdc.ccrs.PlateCarree(), zorder=6)
        ax.text(0, 1.02, panel_title, transform=ax.transAxes, fontsize=10.5, fontweight="semibold", va="bottom")
        ax.text(1, 1.02, f"median share {np.median(share[label]):.0%}; {(share[label] >= 0.8).sum()} of {len(rows)} at ≥ 80 %",
                transform=ax.transAxes, fontsize=8.6, color=ink2, va="bottom", ha="right")
        if q == 1:
            for a_ in (1000, 5000, 25000):
                ax.scatter([], [], s=marker_size(a_), c="#898781", edgecolors=surface, label=f"{a_:,} ha")
            legend = ax.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor=surface, edgecolor=axis,
                                fontsize=8.2, title="Burned area", title_fontsize=8.2, labelspacing=1.1, borderpad=0.9)
            for text in legend.get_texts():
                text.set_color(ink2)
    cax = fig.add_axes([0.30, 0.105, 0.40, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([0, .25, .5, .75, 1])
    cb.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    cb.ax.tick_params(length=0, labelsize=8.5, colors=ink2)
    cb.set_label("Share of other years that were wetter on this date (100 % = driest on record)", fontsize=8.8, color=ink2, labelpad=4)
    fig.text(0.04, 0.015, "See fire/build_fire_catalog.py for sources. Fuel moisture only: LEFIRE has no ignition, wind or spread. Not an exhaustive list of fires.",
              fontsize=8, color=ink3)
    fig.savefig(args.out_dir / "overview_map.png", dpi=150)
    plt.close(fig)

    # ---- timeline: dryness rank of every event over time ----------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(16, 8.4), sharex=True, gridspec_kw={"hspace": 0.16})
    fig.subplots_adjust(left=0.06, right=0.95, top=0.80, bottom=0.07)
    fig.text(0.06, 0.965, "Most big fires start on days when the fuel is unusually dry for the date – but not all of them",
              fontsize=15, fontweight="semibold", va="top")
    fig.text(0.06, 0.925,
              f"Each dot is one fire (n = {len(rows)}): height = share of the other cached years that were wetter on "
              "that date at the fire’s grid cell; size = burned area.", fontsize=9.2, color=ink2, va="top", linespacing=1.5)
    x = date.astype("datetime64[ms]").astype(object)
    for ax, (_, label, panel_title) in zip(axs, [v for v in VARS if v[1] != "fast10h"]):
        ax.axhline(0.5, color=axis, lw=1.0, zorder=0)
        ax.text(1.005, 0.5, "typical", transform=ax.get_yaxis_transform(), fontsize=8.5, color=ink3, va="center")
        ax.scatter(np.array(x)[order], share[label][order], s=marker_size(area[order]) * 1.4, color=blue, alpha=0.55,
                   edgecolors=surface, linewidths=0.8, zorder=3)
        placed = []
        for idx in sorted(np.where(area >= np.percentile(area, 90))[0], key=lambda i: date[i]):
            dy = 6
            for d0, w0, dy0 in placed:
                if abs((date[idx] - d0).astype(int)) < 400 and abs(share[label][idx] - w0) < 0.10 and abs(dy - dy0) < 9:
                    dy = dy0 - 11
            placed.append((date[idx], share[label][idx], dy))
            ax.annotate(rows[idx]["name"].split(",")[0].split(" (")[0], (x[idx], share[label][idx]),
                        xytext=(4, dy), textcoords="offset points", fontsize=7.6, color=ink2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(axis)
            ax.spines[spine].set_linewidth(0.8)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=5)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels(["0 %\nwettest", "25 %", "50 %", "75 %", "100 %\ndriest"])
        ax.text(0, 1.03, panel_title, transform=ax.transAxes, fontsize=10.5, fontweight="semibold", va="bottom")
    axs[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axs[1].set_xlim(date.min().astype("datetime64[Y]").astype(object), (date.max() + np.timedelta64(200, "D")).astype("datetime64[ms]").astype(object))
    fig.text(0.06, 0.012, "Share of other cached years that were wetter on the fire date (ties count half). Fuel moisture only; LEFIRE has no ignition, wind or spread.",
              fontsize=8, color=ink3)
    fig.savefig(args.out_dir / "overview_timeline.png", dpi=150)
    plt.close(fig)
    print(f"wrote {args.out_dir / 'overview_map.png'}, {args.out_dir / 'overview_timeline.png'}, {csv_path}")


if __name__ == "__main__":
    main()
