#!/usr/bin/env python3
"""Check water-balance closure of raw ecLand output, per site and network-wide.

Imported from the sibling plumber2-ecland repo's scripts/check_water_budget.py
and adapted for this repo's gridded run/output/<year>/o_*.nc layout: each year
subdirectory is one "site" (pass --output-dir run/output, or
run/output_depthtrilogy), and within it the closure is computed at every land
grid point. Sea cells (the 1e20 fill value) are masked out; a single-site
PLUMBER2 run is the 1 x 1 case of the same calculation.

For each land point, integrates the terms in o_wat.nc over the whole run
(all positive when water enters the land) and reports the residual of

    Rainf + Snowf + Evap + Qs + Qsb - EWater
      - DelIntercept - DelSoilMoist - DelSWE - DelAquifer

EWater (from o_eva.nc, zero if absent) is the evaporation from ecLand's
open-water tiles -- tile 1 (sea water) and tile 9 (lake). Those are unlimited
reservoirs, so that part of Evap drains no land store and is taken out of the
balance. Without it, the 1988 LIAISE residual follows lake cover (correlation
-0.98) and reaches -53 mm/yr; with it, every point closes within 6.8 mm/yr.

(Qs and Qsb are ADDED here, matching ecLand's own output sign convention --
subtracting them instead manufactures a spurious residual of exactly 2x their
value.) This is the budget /home/pad/check_water_budget.sc has used since 2007,
extended by the -DelAquifer term added alongside the Qrec/Qcap/DelAquifer
diagnostics (ecland e9db58d) so LEGWRECHARGE runs close exactly rather than
legitimately reading the recharge.

DelAquifer, Qrec and Qcap are read if present and silently treated as zero
otherwise, so this runs unchanged against output from a build that predates
those diagnostics (LEGWRECHARGE off, or an older ecLand). When they are
present, also reports "clamp loss" = (Qrec-Qcap)-DelAquifer: nonzero only
where the water-table's RGWTD_MIN/RDBEDROCK clamp overrides the raw
recharge/capillary-rise flux without a matching correction to what was
diverted from Qsb -- see ecland 2470a28 for the cold-start case this
diagnostic caught, and the commit's note on the smaller within-run case that
is not yet fixed.

Exits non-zero if any land point's residual exceeds --tol-frac of its
precipitation (default 1%) AND --tol-abs in absolute terms (default 5 mm/yr)
-- both must be exceeded, so a bone-dry point's tiny precipitation total does
not trip the check on a fraction of nearly nothing -- or if the residual is NaN
at any land point (a NaN never counts as a pass). The per-site table
(--out-csv) gives land-point means plus the worst point and its lat/lon;
--per-point-csv writes the residual at every land point. Intended to run right after a raw
ecLand run, before or alongside postproc_plumber2.py -- it reads o_wat.nc
directly, and DelSoilMoist/DelSWE/DelIntercept/Qrec/Qcap/DelAquifer do not
survive into the postprocessed PLUMBER2-schema files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

DEFAULT_OUTPUT_DIR = Path('output')
SECONDS_PER_YEAR = 365.25 * 86400.0


def _read(ds: Dataset, name: str) -> np.ndarray:
    # float64, with fill-valued (sea) cells as NaN; a genuine NaN in the data stays NaN
    return np.ma.filled(ds.variables[name][:].astype('f8'), np.nan)


def _coords(ds: Dataset, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    # per-point lat/lon on the (lat, lon) grid; NaN when the file carries no coordinates
    if 'lat' in ds.variables and 'lon' in ds.variables:
        lat, lon = np.asarray(ds.variables['lat'][:], 'f8'), np.asarray(ds.variables['lon'][:], 'f8')
        if lat.ndim == 1 and (lat.size, lon.size) == shape:
            return tuple(np.meshgrid(lat, lon, indexing='ij'))
    return np.full(shape, np.nan), np.full(shape, np.nan)


def read_site(site_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    """Closure per grid point; returns the site summary and the per-point table.

    Every term keeps its (lat, lon) shape. Land points are the cells that do
    not hold the fill value in Qs, so sea cells never enter the arithmetic;
    a NaN at a land point propagates to that point's residual and fails the check.
    """
    wat_path = site_dir / 'o_wat.nc'
    if not wat_path.exists():
        return None
    with Dataset(wat_path) as ds:
        if 'Qs' not in ds.variables:
            return None
        time = np.asarray(ds.variables['time'][:])
        dt = float(np.median(np.diff(time))) if len(time) > 1 else 1800.0
        n = ds.variables['Qs'].shape[0]
        years = n * dt / SECONDS_PER_YEAR
        land = ~np.ma.getmaskarray(ds.variables['Qs'][0])
        lat, lon = _coords(ds, land.shape)

        def flux_total(name: str) -> np.ndarray:
            # kg m-2 s-1, summed and converted to a run-total in kg m-2 (mm), per point
            return np.sum(_read(ds, name), axis=0) * dt if name in ds.variables else np.zeros(land.shape)

        def store_total(name: str) -> np.ndarray:
            # kg m-2 per step already -- summed directly, no dt factor
            return np.sum(_read(ds, name), axis=0) if name in ds.variables else np.zeros(land.shape)

        rainf, snowf = flux_total('Rainf'), flux_total('Snowf')
        evap, qs, qsb = flux_total('Evap'), flux_total('Qs'), flux_total('Qsb')
        qrec, qcap = flux_total('Qrec'), flux_total('Qcap')
        dsm, dswe, dint = store_total('DelSoilMoist'), store_total('DelSWE'), store_total('DelIntercept')
        daq = store_total('DelAquifer')
        has_aquifer_diag = 'Qrec' in ds.variables and 'DelAquifer' in ds.variables

    # Open-water evaporation (tiles 1 + 9: sea water and lakes) is part of Evap but is
    # drawn from ecLand's unlimited water body, not from any land store above.
    ewater = np.zeros(land.shape)
    eva_path = site_dir / 'o_eva.nc'
    if eva_path.exists():
        with Dataset(eva_path) as ds:
            if 'EWater' in ds.variables:
                ewater = np.sum(_read(ds, 'EWater'), axis=0) * dt

    precip = rainf + snowf
    residual = precip + evap + qs + qsb - ewater - dint - dsm - dswe - daq

    points = pd.DataFrame({
        'site': site_dir.name,
        'lat': lat[land],
        'lon': lon[land],
        'precip_mm_yr': precip[land] / years,
        'residual_mm_yr': residual[land] / years,
    })
    res, p = points.residual_mm_yr.to_numpy(), points.precip_mm_yr.to_numpy()
    n_nan = int(np.isnan(res).sum())
    worst = int(np.nanargmax(np.abs(res))) if n_nan < res.size else None
    summary = {
        'site': site_dir.name,
        'years': years,
        'n_points': int(land.sum()),
        'precip_mm_yr': float(p.mean()),
        'ewater_mm_yr': float(np.mean(ewater[land])) / years,
        'residual_mm_yr': float(res.mean()),  # land-point mean; NaN if any point is NaN
        'residual_pct_of_P': 100.0 * float(res.mean()) / float(p.mean()) if p.mean() else float('nan'),
        'worst_residual_mm_yr': res[worst] if worst is not None else float('nan'),
        'worst_lat': points.lat.iloc[worst] if worst is not None else float('nan'),
        'worst_lon': points.lon.iloc[worst] if worst is not None else float('nan'),
        'n_nan_points': n_nan,
        'clamp_loss_mm_yr': float(np.mean((qrec - qcap - daq)[land])) / years if has_aquifer_diag else float('nan'),
        'has_aquifer_diag': has_aquifer_diag,
    }
    return summary, points


def discover_sites(output_dir: Path, site_filter: set[str] | None) -> list[Path]:
    dirs = sorted(p for p in output_dir.iterdir() if p.is_dir() and (p / 'o_wat.nc').exists())
    if site_filter:
        dirs = [d for d in dirs if d.name.split('_')[0] in site_filter or d.name in site_filter]
    return dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                    help='Raw ecLand output root, one subdirectory per site (default: output/).')
    p.add_argument('--out-csv', type=Path, default=None,
                    help='Optional path to write the per-site table as CSV.')
    p.add_argument('--per-point-csv', type=Path, default=None,
                    help='Optional path to write the residual at every land point as CSV (for maps).')
    p.add_argument('--site', action='append', default=None, help='Optional site filter; repeatable.')
    p.add_argument('--tol-frac', type=float, default=1.0,
                    help='Residual tolerance as %% of precipitation (default: 1.0).')
    p.add_argument('--tol-abs', type=float, default=5.0,
                    help='Residual tolerance in mm/yr, combined with --tol-frac (default: 5.0).')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    site_filter = set(args.site) if args.site else None
    site_dirs = discover_sites(args.output_dir, site_filter)
    if not site_dirs:
        print(f'No site output found under {args.output_dir}')
        return 2

    results = [r for d in site_dirs if (r := read_site(d)) is not None]
    points = pd.concat([p for _, p in results], ignore_index=True)
    df = pd.DataFrame([s for s, _ in results])
    # a point fails if it is NaN, or beyond BOTH tolerances; NaN must never read as PASS
    within = ((points.residual_mm_yr.abs() <= args.tol_abs)
              | (100.0 * points.residual_mm_yr.abs() / points.precip_mm_yr.abs() <= args.tol_frac))
    df['n_failing_points'] = df.site.map((~within).groupby(points.site).sum()).astype(int)
    df = df.sort_values('worst_residual_mm_yr', key=lambda s: s.abs(), ascending=False)

    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', lambda x: f'{x:9.4f}')
    print(df.to_string(index=False))
    print()
    print(f'{len(df)} sites, {len(points)} land points.  |residual| per point: '
          f'mean={points.residual_mm_yr.abs().mean():.4f}  median={points.residual_mm_yr.abs().median():.4f}  '
          f'max={points.residual_mm_yr.abs().max():.4f} mm/yr')
    n_diag = int(df.has_aquifer_diag.sum())
    if n_diag:
        print(f'{n_diag} sites carry Qrec/Qcap/DelAquifer.  |clamp loss| among them: '
              f'mean={df.loc[df.has_aquifer_diag, "clamp_loss_mm_yr"].abs().mean():.4f}  '
              f'max={df.loc[df.has_aquifer_diag, "clamp_loss_mm_yr"].abs().max():.4f} mm/yr')

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f'Wrote {args.out_csv}')
    if args.per_point_csv:
        args.per_point_csv.parent.mkdir(parents=True, exist_ok=True)
        points.to_csv(args.per_point_csv, index=False)
        print(f'Wrote {args.per_point_csv}')

    failing = df[df.n_failing_points > 0]
    if len(failing):
        print(f'\nFAIL: {len(failing)} site(s) have land points beyond both {args.tol_abs} mm/yr '
              f'and {args.tol_frac}% of P, or NaN:')
        print(failing[['site', 'n_failing_points', 'n_nan_points', 'worst_residual_mm_yr',
                       'worst_lat', 'worst_lon']].to_string(index=False))
        return 1
    print(f'\nPASS: all land points within {args.tol_abs} mm/yr or {args.tol_frac}% of P.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
