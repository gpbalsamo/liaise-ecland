#!/usr/bin/env python3
"""Per-event figure: how dry was the fuel on a documented fire's ignition day?

For each event in the catalogue (fire/build_fire_catalog.py), draws three maps on the
ignition date -- fast dead fuel (10 h class, absolute), slow dead fuel (1000 h class,
dryness rank) and live fuel moisture for low vegetation (dryness rank) -- plus two time
series at the fire's own grid cell (the event year against the other cached years' 10-90 %
band). "Dryness rank" = the share of the other cached years that were wetter on that same
calendar date (100 % = the driest that date has been across the whole cache).

The grid this runs on is coarse (0.5 deg / ~55 km in the committed LIAISE setup), so any
fire in the catalogue is far smaller than one cell: these figures show the regional
fuel-dryness *context* on the day, not the fire itself. LEFIRE has no ignition, wind or
fire-spread model, so nothing here predicts whether or where a fire starts.

Usage
-----
    # one event
    python3 plot_fire_event_maps.py --catalog fire/data/catalog.json \\
        --cache fire/work/fire_daily_moisture.npz --out-dir fire/work/plots \\
        --keys 2023-04-16_cerbere-banyuls-fire-pyrenees-orientales-fr

    # every event in the catalogue, plus a combined stats JSON
    python3 plot_fire_event_maps.py --catalog fire/data/catalog.json \\
        --cache fire/work/fire_daily_moisture.npz --out-dir fire/work/plots \\
        --stats-out fire/work/stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_dryness_common as fdc

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": fdc.SURFACE, "axes.facecolor": fdc.SURFACE,
    "savefig.facecolor": fdc.SURFACE, "text.color": fdc.INK, "axes.labelcolor": fdc.INK2,
    "xtick.color": fdc.INK2, "ytick.color": fdc.INK2, "font.size": 9,
})

# per-cache-variable "tied" tolerance -- DFMC_1/10 sit at a hard 0.30 cap and LFMC saturates too,
# so an exact-equality dryness rank would be noisy; values within EPS of each other count as tied.
EPS = {"DFMC_10_min": 5e-4, "DFMC_1000_mean": 5e-4, "LFMC_L_mean": 0.05}

CONTEXT_WINDOW = np.arange(-150, 31)  # days before/after ignition shown in the time-series panels


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=Path("fire/data/catalog.json"))
    p.add_argument("--cache", type=Path, required=True, help="output of extract_fire_daily_moisture.py")
    p.add_argument("--out-dir", type=Path, default=Path("fire/work/plots"))
    p.add_argument("--stats-out", type=Path, default=None,
                    help="write per-event dryness stats here as JSON (default: <out-dir>/stats.json)")
    p.add_argument("--keys", nargs="*", default=None, help="process only these event keys (default: all)")
    return p.parse_args()


class DrynessMapper:
    def __init__(self, cache: Path):
        self.z = np.load(cache)
        self.dates = self.z["dates"].astype("datetime64[D]")
        self.lat, self.lon, self.land = self.z["lat"], self.z["lon"], self.z["land"]
        self.lon_edges, self.lat_edges = fdc.cell_edges(self.lon), fdc.cell_edges(self.lat)
        self.extent = [self.lon_edges.min(), self.lon_edges.max(), self.lat_edges.min(), self.lat_edges.max()]
        self.years = np.unique(self.dates.astype("datetime64[Y]").astype(int) + 1970)

    def make_event(self, ev: dict, out_dir: Path) -> dict:
        year, month_day = int(ev["date"][:4]), ev["date"][5:]
        t0 = fdc.day_index(self.dates, year, month_day)
        if t0 is None:
            raise ValueError(f"{ev['key']}: ignition date not in the cached range")
        other_idx = [i for y in self.years if y != year and (i := fdc.day_index(self.dates, int(y), month_day)) is not None]
        j, i = fdc.nearest_land_cell(self.lat, self.lon, self.land, ev["lat"], ev["lon"])

        fields = {}
        for var in ("DFMC_10_min", "DFMC_1000_mean", "LFMC_L_mean"):
            fields[var] = dict(event=self.z[var][t0], other_years=self.z[var][other_idx])

        fig = plt.figure(figsize=(16, 9.6))
        gs = fig.add_gridspec(2, 6, height_ratios=[1.55, 1], hspace=0.42, wspace=0.40,
                               left=0.05, right=0.985, top=0.835, bottom=0.075)
        title_date = np.datetime64(ev["date"]).astype(object)
        fig.text(0.05, 0.965, f"{ev['name']}, {title_date:%-d %B %Y}: how dry was the fuel?",
                  fontsize=15, fontweight="semibold", va="top")
        cell_is_land = self.land[int(np.abs(self.lat - ev["lat"]).argmin()), int(np.abs(self.lon - ev["lon"]).argmin())]
        moved_note = "" if cell_is_land else (
            f" The grid cell containing the fire is sea in the model, so the nearest land cell "
            f"({self.lat[j]:.2f}°N, {self.lon[i]:.2f}°E) is used.")
        fig.text(0.05, 0.925,
                  f"Burned {ev['size']}; {ev['extra']}. Grid cells are far larger than the fire itself.  "
                  f"● ignition site, □ grid cell used below.\n"
                  f"Rank maps compare the day with the same date in the {len(other_idx)} other cached years "
                  f"(red = drier than usual).{moved_note}",
                  fontsize=9.2, color=fdc.INK2, va="top", linespacing=1.5)

        rank_label = "Share of other years that were wetter on this date (100 % = driest on record)"
        panels = [
            ("DFMC_10_min", "Fast dead fuel (10 h), driest hour of the day", "value", fdc.SEQ_BLUE, Normalize(0, 0.30),
             "Moisture (fraction)", lambda v: f"{v:.3f}"),
            ("DFMC_1000_mean", "Slow dead fuel (1000 h): how unusual?", "rank", fdc.DIVERGING_DRY_WET.reversed(),
             Normalize(0, 1), rank_label, lambda v: f"{v:.3f}"),
            ("LFMC_L_mean", "Live fuel moisture, low vegetation: how unusual?", "rank", fdc.DIVERGING_DRY_WET.reversed(),
             Normalize(0, 1), rank_label, lambda v: f"{v:.1f} %"),
        ]
        stats = {}
        for p, (var, panel_title, mode, cmap, norm, colorbar_label, fmt) in enumerate(panels):
            f = fields[var]
            grid = f["event"] if mode == "value" else fdc.wetter_share(f["event"], f["other_years"], EPS[var])
            cell_other_years = f["other_years"][:, j, i]
            cell_other_years = cell_other_years[np.isfinite(cell_other_years)]
            value = float(f["event"][j, i])
            drier = int((cell_other_years > value + EPS[var]).sum())
            tied = int((np.abs(cell_other_years - value) <= EPS[var]).sum())
            stats[var] = dict(value=value, other_years_mean=float(cell_other_years.mean()), n_other_years=len(cell_other_years),
                               drier_than_n=drier, tied=tied, dryness_share=(drier + 0.5 * tied) / len(cell_other_years))

            ax = fig.add_subplot(gs[0, 2 * p:2 * p + 2], projection=fdc.ccrs.PlateCarree())
            im = ax.pcolormesh(self.lon_edges, self.lat_edges, np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                                transform=fdc.ccrs.PlateCarree(), zorder=2, shading="flat", rasterized=True)
            fdc.draw_basemap(ax, self.extent, with_cities=(p == 0))
            ax.add_patch(plt.Rectangle((self.lon_edges[i], self.lat_edges[j]),
                                        self.lon_edges[i + 1] - self.lon_edges[i], self.lat_edges[j + 1] - self.lat_edges[j],
                                        fill=False, edgecolor=fdc.INK, lw=1.3, transform=fdc.ccrs.PlateCarree(), zorder=5))
            ax.plot(ev["lon"], ev["lat"], "o", ms=7, mfc=fdc.INK, mec=fdc.SURFACE, mew=1.6,
                    transform=fdc.ccrs.PlateCarree(), zorder=7)
            ax.text(0, 1.085, panel_title, transform=ax.transAxes, fontsize=10.5, fontweight="semibold", va="bottom")
            ax.text(0, 1.02, fdc.rank_text(value, cell_other_years, fmt, EPS[var]), transform=ax.transAxes,
                    fontsize=8.6, color=fdc.INK2, va="bottom")
            cax = ax.inset_axes([0.06, -0.135, 0.88, 0.035])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            if mode == "rank":
                cb.set_ticks([0, .25, .5, .75, 1])
                cb.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
                cb.ax.tick_params(length=0, labelsize=8, colors=fdc.INK2)
            cb.set_label(colorbar_label, fontsize=7.8 if mode == "rank" else 8.4, color=fdc.INK2, labelpad=3)

        def context_series(var: str, y: int) -> np.ndarray:
            t = fdc.day_index(self.dates, y, month_day)
            out = np.full(len(CONTEXT_WINDOW), np.nan)
            if t is None:
                return out
            idx = t + CONTEXT_WINDOW
            valid = (idx >= 0) & (idx < len(self.dates))
            out[valid] = self.z[var][idx[valid], j, i]
            return out

        xs = (np.datetime64(ev["date"]) + CONTEXT_WINDOW).astype("datetime64[ms]").astype(object)
        context_panels = [("DFMC_1000_mean", "Slow dead fuel (1000 h) at the fire cell", "fraction"),
                           ("LFMC_L_mean", "Live fuel moisture (low vegetation) at the fire cell", "%")]
        for q, (var, panel_title, unit) in enumerate(context_panels):
            ax = fig.add_subplot(gs[1, 3 * q:3 * q + 3])
            event_series = context_series(var, year)
            other_series = np.array([context_series(var, int(y)) for y in self.years if y != year])
            p10, p50, p90 = np.nanpercentile(other_series, [10, 50, 90], axis=0)
            ax.fill_between(xs, p10, p90, color="#898781", alpha=0.20, lw=0, label=f"10–90 % of {len(other_series)} other years")
            ax.plot(xs, p50, color="#898781", lw=1.2, label="Median of other years")
            ax.plot(xs, event_series, color=fdc.BLUE, lw=2.0, solid_capstyle="round", label=str(year))
            ignite_x = np.datetime64(ev["date"]).astype("datetime64[ms]").astype(object)
            ax.axvline(ignite_x, color=fdc.INK, lw=1.0, zorder=0)
            ax.text(ignite_x, 1.0, " ignition", transform=ax.get_xaxis_transform(), fontsize=8.5, color=fdc.INK2, va="top")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(fdc.AXIS)
                ax.spines[spine].set_linewidth(0.8)
            ax.grid(axis="y", color=fdc.GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0, pad=5)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylabel(unit, color=fdc.INK2)
            ax.text(0, 1.16, panel_title, transform=ax.transAxes, fontsize=10.5, fontweight="semibold", va="bottom")
            legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.6,
                                handlelength=1.6, columnspacing=1.4)
            for text in legend.get_texts():
                text.set_color(fdc.INK2)

        fig.text(0.05, 0.015,
                  f"Event date and size: {ev['src']}. Fuel moisture only: LEFIRE has no ignition, wind or fire spread. "
                  "Other years matched by month-day; ties (cap saturation) count half; time series show the fire cell only.",
                  fontsize=8, color=fdc.INK3)
        out_path = out_dir / f"fire_map_{ev['key']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return dict(stats=stats, cell_lat=float(self.lat[j]), cell_lon=float(self.lon[i]),
                    moved_to_land=bool(not cell_is_land), n_other_years=len(other_idx), file=str(out_path))


def main() -> None:
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_out = args.stats_out or args.out_dir / "stats.json"

    catalog = json.load(args.catalog.open())
    if args.keys:
        wanted = set(args.keys)
        catalog = [e for e in catalog if e["key"] in wanted]
        missing = wanted - {e["key"] for e in catalog}
        if missing:
            sys.exit(f"unknown event key(s): {sorted(missing)}")

    mapper = DrynessMapper(args.cache)
    results, t0 = {}, time.time()
    for n, ev in enumerate(catalog, 1):
        try:
            results[ev["key"]] = mapper.make_event(ev, args.out_dir)
        except Exception as e:  # noqa: BLE001 -- keep going, report at the end
            results[ev["key"]] = {"error": repr(e)}
            print(f"ERROR {ev['key']}: {e}", file=sys.stderr)
            traceback.print_exc()
        print(f"{n}/{len(catalog)} {ev['key']} ({time.time() - t0:.0f}s)", flush=True)

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, stats_out.open("w"))
    n_errors = sum("error" in v for v in results.values())
    print(f"done: {len(results)} events, {n_errors} errors -> {stats_out}")


if __name__ == "__main__":
    main()
