#!/usr/bin/env python3
"""Compact daily cache of a finished LEFIRE run's fuel-moisture fields, for fast event lookups.

Reduces years of hourly `o_fire.nc` (one per year, under `--run-root/output/<year>/`) to a
single small NetCDF-free `.npz` of per-cell daily statistics, so the event-comparison scripts
don't need to re-open dozens of large hourly files for every fire looked up.

`o_fire.nc`'s own `lat`/`lon` variables are never written by ecLand (checked: always the
NetCDF fill value) -- the grid is instead read from `--surfclim`, which also supplies the
land/sea mask used to blank sea cells.

Usage
-----
    python3 extract_fire_daily_moisture.py --run-root /path/to/run_1988_2024 \\
        --surfclim init_clim/work/surfclim --start-year 1988 --end-year 2024 \\
        --out fire/work/fire_daily_moisture.npz
"""
from __future__ import annotations

import argparse
import functools
import re
from pathlib import Path

import netCDF4 as nc
import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001 -- always flush under sbatch

# name in the cache -> (o_fire.nc variable, per-day reduction)
DAILY_STATS = {
    "DFMC_1_min": ("DFMC_1", np.nanmin),
    "DFMC_10_min": ("DFMC_10", np.nanmin),
    "DFMC_100_mean": ("DFMC_100", np.nanmean),
    "DFMC_1000_mean": ("DFMC_1000", np.nanmean),
    "LFMC_L_mean": ("LFMC_L", np.nanmean),
    "LFMC_H_mean": ("LFMC_H", np.nanmean),
}


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True,
                    help="a run_liaise_ecland.sh RUN_ROOT with LEFIRE on (output/<year>/o_fire.nc per year)")
    p.add_argument("--surfclim", type=Path, default=Path("init_clim/work/surfclim"))
    p.add_argument("--start-year", type=int, default=1988)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--out", type=Path, default=Path("fire/work/fire_daily_moisture.npz"))
    return p.parse_args()


def main() -> None:
    args = get_args()
    with nc.Dataset(args.surfclim) as s:
        lat = np.ma.filled(s["lat"][:], np.nan).astype("f8")
        lon = np.ma.filled(s["lon"][:], np.nan).astype("f8")
        land = np.ma.filled(s["Mask"][:], 0) > 0

    out = {k: [] for k in DAILY_STATS}
    all_days = []
    for year in range(args.start_year, args.end_year + 1):
        f = args.run_root / "output" / str(year) / "o_fire.nc"
        d = nc.Dataset(f)
        d.set_auto_mask(False)
        t = d["time"][:]
        base = np.datetime64(re.search(r"since (\d{4}-\d{2}-\d{2})", d["time"].units).group(1))
        n_hours = len(t) - 1  # drop the shared year-boundary endpoint record (see CLAUDE.md)
        if n_hours % 24 != 0:
            raise ValueError(f"{f}: {n_hours} hours after dropping the endpoint is not a whole number of days")
        days = (base + np.asarray(t[:n_hours:24], "f8").astype("timedelta64[s]")).astype("datetime64[D]")
        all_days.append(days)
        for cache_name, (var, reduce_fn) in DAILY_STATS.items():
            x = d[var][:n_hours].astype("f4")
            x[:, ~land] = np.nan
            x[np.abs(x) > 1e19] = np.nan
            if var.startswith("LFMC"):
                x[0] = np.nan  # diagnostic buffer is 0 at the initial-state record, not real data
            daily = reduce_fn(x.reshape(n_hours // 24, 24, *x.shape[1:]), axis=1)
            out[cache_name].append(daily.astype("f4"))
        print(f"{year}: {len(days)} days")

    dates = np.concatenate(all_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, dates=dates.astype("datetime64[D]").astype("int64"), lat=lat, lon=lon, land=land,
                        **{k: np.concatenate(v) for k, v in out.items()})
    print(f"wrote {args.out} ({dates[0]} to {dates[-1]}, {len(dates)} days)")


if __name__ == "__main__":
    main()
