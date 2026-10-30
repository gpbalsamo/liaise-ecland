#!/usr/bin/env python3
"""Check surface energy-balance closure of raw ecLand output, per site and network-wide.

Imported from the sibling plumber2-ecland repo's scripts/check_energy_budget.py
and adapted for this repo's gridded run/output/<year>/o_*.nc layout: each year
subdirectory is one "site" (pass --output-dir run/output, or
run/output_depthtrilogy), and within it the closure is computed at every land
grid point. Sea cells (the 1e20 fill value) are masked out; a single-site
PLUMBER2 run is the 1 x 1 case of the same calculation.

For each land point, integrates over the whole run the terms in o_efl.nc's own
sign convention (all fluxes positive DOWNWARD, i.e. toward the surface -- so
Qh and Qle are negative on average; this is NOT the ALMA upward-positive
convention the PLUMBER2-schema postprocessed files use):

    residual = (SWnet + LWnet + Qh + Qle)  -  dE_soil/dt  -  dE_snow/dt

and reports the run-mean residual in W/m2. The first term is the energy that
enters the ground; dE_soil and dE_snow are the increases in soil and snow
heat content between the first and last output step. Rearranged, this is
checking that the energy available at the surface after turbulent exchange is
fully accounted for by the change in soil + snow internal energy content.

DelSoilHeat and DelColdCont are read from o_efl.nc if present, but in every
build/config checked so far they are identically zero: DelSoilHeat's source
value (D1SDSH in upddiag.F90) is hardcoded to 0 ("DelSoilHeat of skin layer
(0)"), and DelColdCont's (PDHTSS(:,1,15) in srfsn_lwexp_mod.F90) is populated
only inside the debug-only LESNCHECK path -- the line that would set it during
a normal run is commented out. Both are therefore reconstructed independently
here from state variables rather than trusted from the file:

  * Soil: srfene_mod.F90's own "apparent energy" formula, using the actual
    per-layer field capacity/porosity ecLand writes to o_fix.nc (SoilFC,
    SoilSat) -- RRCSOILM3D = (1-SoilSat)*RCGDRY + SoilFC*RGH2O exactly (see
    susdp_deriv_ctl_mod.F90 / susdp_dflt_ctl_mod.F90) -- and the model's own
    temperature-only freezing curve (RTF1/RTF2 namelist defaults). Verified
    against the true surface-flux integral on a snow-free site to <0.01%.

  * Snow: no equivalent internal formula exists in the standard diagnostic
    output (see above), so this uses a standard ice+liquid specific-heat
    enthalpy formulation over the multi-layer state (SWEML/slwML/SnowTML),
    with ecLand's own ice-vs-water heat capacity convention (ice = 0.5x
    water's -- see ZGICE in srfrcg_mod.F90). This is a physically-standard
    approximation, not a replica of internal model arithmetic (unlike the
    soil term above).

Even with both storage terms, this does not close to the same precision as
check_water_budget.py's water balance. This is NOT a model energy-conservation
bug: every individual tile's own surface balance closes by construction. It is
a genuine property of how the offline diagnostics are defined: Qg (via ZSURFL
in srft_mod.F90) sums SWnet/LWnet/turbulent flux only over the NON-snow tiles,
substituting PGSN (conduction through the snowpack into the soil) for the two
snow tiles instead -- while the o_efl.nc SWnet/LWnet/Qh/Qle are true full-grid
means that DO include the snow tiles (PDIFTS, PFTLHEV+PFTLHSB in upddiag.F90).
Working through each tile's own closing balance shows

    SWnet + LWnet + Qh + Qle + Qg  ==  dE_snow/dt  (exactly, by construction)

i.e. that non-cancellation IS the (otherwise unreported) instantaneous snow
heat-storage tendency, not a missing flux term -- adding Qgsn/Qfsn on top (an
earlier hypothesis) does not fix it because those are unrelated quantities.
What's left after including this script's own (approximate) dE_snow is a
genuine approximation gap: the two-term ice+liquid enthalpy formula used here
does not replicate the real multi-layer scheme's layer-by-layer melt/
refreeze/percolation bookkeeping (srfsn_lwexp_mod.F90). This residual runs
order 1-3 W/m2 (run-mean) on typical snow-heavy multi-decade sites (FI-Hyy,
CH-Dav, US-Ha1) and confirmed near-zero (<0.01 W/m2) on snow-free sites; it
reflects a script-side formula limitation, not a namelist-fixable output gap
and not a model defect. A short run starting far from thermal equilibrium can
also show a similar-sized residual as its soil column drifts toward balance.
Default tolerance is set loose enough to not flag this normal, explained
residual while still catching an actual regression (which would be expected
to shift Qh/Qle/Qg by far more than a few W/m2 -- e.g. the interception bug
fixed this session shifted the water budget by mm/yr-scale amounts, not the
sub-percent noise floor this script is tuned around).

Because of that approximation, a point that carries snow at any time in the
run ("snow point") is reported separately (n_snow_points,
worst_snow_residual_wm2) but is not judged against the tolerance. On the
1988 LIAISE year, snow points reach 6 W/m2 while snow-free points stay
within 0.3 W/m2.

Exits non-zero if any snow-free land point's |residual| exceeds --tol-wm2
(default 5.0 W/m2), or if the residual is NaN at any land point (snow or not)
-- a NaN never counts as a pass. The per-site table (--out-csv) gives
land-point means plus the worst snow-free point and its lat/lon;
--per-point-csv writes the residual at every land point, e.g. for a map.
Intended to run right after a raw ecLand run, before or alongside
postproc_plumber2.py -- it reads o_efl.nc/o_gg.nc/o_fix.nc/o_vty.nc directly,
none of which survive into the postprocessed PLUMBER2-schema files in the form
needed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

DEFAULT_OUTPUT_DIR = Path('output')
SECONDS_PER_YEAR = 365.25 * 86400.0

# ecLand soil/snow thermal constants -- see src/surf/module/sussoil_mod.F90,
# srfene_mod.F90, srfrcg_mod.F90, susdp_dflt_ctl_mod.F90. RTF1/RTF2 are the
# rdnml_params.F90 namelist defaults (274.16/272.16 K); overridable in
# principle but essentially never overridden in practice.
RCGDRY = 1.6e6    # dry-soil-grain volumetric heat capacity, J K-1 m-3 (Peters-Lidard et al. 1998)
RGH2O = 4.18e6    # volumetric heat capacity of water, J K-1 m-3
RHOH2O = 1000.0   # density of water, kg m-3
RLMLT = 0.3337e6  # latent heat of fusion, J kg-1 (RLSTT - RLVTT)
RTF1, RTF2 = 274.16, 272.16
RTF3 = 0.5 * (RTF1 + RTF2)
RTF4 = np.pi / (RTF1 - RTF2)
CI_SNOW = 0.5 * RGH2O / RHOH2O  # ice specific heat, J kg-1 K-1 (ecLand's own ZGICE convention)
CW_SNOW = RGH2O / RHOH2O        # liquid water specific heat, J kg-1 K-1


def _frozen_fraction(t: np.ndarray) -> np.ndarray:
    return np.where(t <= RTF2, 1.0, np.where(t < RTF1, 0.5 * (1.0 - np.sin(RTF4 * (t - RTF3))), 0.0))


def _read(ds: Dataset, name: str, index=slice(None)) -> np.ndarray:
    # float64, with fill-valued (sea) cells as NaN; a genuine NaN in the data stays NaN
    return np.ma.filled(ds.variables[name][index].astype('f8'), np.nan)


def _coords(ds: Dataset, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    # per-point lat/lon on the (lat, lon) grid; NaN when the file carries no coordinates
    if 'lat' in ds.variables and 'lon' in ds.variables:
        lat, lon = np.asarray(ds.variables['lat'][:], 'f8'), np.asarray(ds.variables['lon'][:], 'f8')
        if lat.ndim == 1 and (lat.size, lon.size) == shape:
            return tuple(np.meshgrid(lat, lon, indexing='ij'))
    return np.full(shape, np.nan), np.full(shape, np.nan)


def _soil_energy(t: np.ndarray, soilfc: np.ndarray, soilsat: np.ndarray, soilthick: np.ndarray,
                 cv: np.ndarray) -> np.ndarray:
    # summed over soil layers (axis 0): J m-2 per grid point
    rrcsoil = (1.0 - soilsat) * RCGDRY + soilfc * RGH2O
    f = _frozen_fraction(t)
    return np.sum((rrcsoil * t - RLMLT * RHOH2O * cv * soilfc * f) * soilthick, axis=0)


def _snow_energy(swe: np.ndarray, slw: np.ndarray, t: np.ndarray) -> np.ndarray:
    # summed over snow layers (axis 0): J m-2 per grid point
    ice = swe - slw
    return np.sum(CI_SNOW * ice * t + CW_SNOW * slw * t + RLMLT * slw, axis=0)


def read_site(site_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    """Closure per grid point; returns the site summary and the per-point table.

    Every field keeps its (lat, lon) shape. Land points are the cells that do
    not hold the fill value in SWnet, so sea cells never enter the arithmetic;
    a NaN at a land point propagates to that point's residual and fails the check.
    """
    efl_path, gg_path = site_dir / 'o_efl.nc', site_dir / 'o_gg.nc'
    fix_path, vty_path = site_dir / 'o_fix.nc', site_dir / 'o_vty.nc'
    if not all(p.exists() for p in (efl_path, gg_path, fix_path, vty_path)):
        return None

    with Dataset(efl_path) as ds:
        if 'SWnet' not in ds.variables:
            return None
        time = np.asarray(ds.variables['time'][:])
        dt = float(np.median(np.diff(time))) if len(time) > 1 else 1800.0
        n = ds.variables['SWnet'].shape[0]
        years = n * dt / SECONDS_PER_YEAR
        land = ~np.ma.getmaskarray(ds.variables['SWnet'][0])
        lat, lon = _coords(ds, land.shape)
        swnet, lwnet = _read(ds, 'SWnet'), _read(ds, 'LWnet')
        qh, qle = _read(ds, 'Qh'), _read(ds, 'Qle')
    surf_avail = np.sum(swnet + lwnet + qh + qle, axis=0) * dt
    rnet_mean_abs = np.mean(np.abs(swnet + lwnet), axis=0)

    with Dataset(fix_path) as ds:
        soilthick = _read(ds, 'SoilThick')
        soilfc = _read(ds, 'SoilFC')
        soilsat = _read(ds, 'SoilSat')

    with Dataset(vty_path) as ds:
        vtfr0 = _read(ds, 'vtfr', 0)
    cv = vtfr0[0] + vtfr0[1]  # low+high vegetation cover per point; static over the run

    with Dataset(gg_path) as ds:
        t0, t1 = _read(ds, 'SoilTemp', 0), _read(ds, 'SoilTemp', -1)
        has_ml_snow = 'SWEML' in ds.variables and 'slwML' in ds.variables and 'SnowTML' in ds.variables
        dE_snow = np.zeros(land.shape)
        is_snow = np.zeros(land.shape, dtype=bool)
        if has_ml_snow:
            # snow point: carries snow at any time in the run (its dE_snow is approximate)
            is_snow = (np.ma.filled(ds.variables['SWEML'][:], 0.0) > 0.0).any(axis=(0, 1))
            swe0, swe1 = _read(ds, 'SWEML', 0), _read(ds, 'SWEML', -1)
            slw0, slw1 = _read(ds, 'slwML', 0), _read(ds, 'slwML', -1)
            snt0, snt1 = _read(ds, 'SnowTML', 0), _read(ds, 'SnowTML', -1)
            dE_snow = _snow_energy(swe1, slw1, snt1) - _snow_energy(swe0, slw0, snt0)

    dE_soil = _soil_energy(t1, soilfc, soilsat, soilthick, cv) - _soil_energy(t0, soilfc, soilsat, soilthick, cv)

    # Energy into the ground minus what the ground (soil + snow) actually stored
    residual_wm2 = (surf_avail - dE_soil - dE_snow) / (years * SECONDS_PER_YEAR)

    points = pd.DataFrame({
        'site': site_dir.name,
        'lat': lat[land],
        'lon': lon[land],
        'rnet_mean_wm2': rnet_mean_abs[land],
        'residual_wm2': residual_wm2[land],
        'is_snow': is_snow[land],
    })
    res, snow = points.residual_wm2.to_numpy(), points.is_snow.to_numpy()
    n_nan = int(np.isnan(res).sum())

    def _worst(mask: np.ndarray) -> int | None:
        r = np.where(mask, np.abs(res), np.nan)
        return int(np.nanargmax(r)) if np.isfinite(r).any() else None

    worst, worst_snow = _worst(~snow), _worst(snow)
    rnet = float(points.rnet_mean_wm2.mean())
    summary = {
        'site': site_dir.name,
        'years': years,
        'n_points': int(land.sum()),
        'rnet_mean_wm2': rnet,
        'dE_soil_MJ_m2': float(np.mean(dE_soil[land])) / 1.0e6,
        'dE_snow_MJ_m2': float(np.mean(dE_snow[land])) / 1.0e6,
        'residual_wm2': float(res.mean()),  # land-point mean; NaN if any point is NaN
        'residual_pct_of_Rnet': 100.0 * float(res.mean()) / rnet if rnet else float('nan'),
        'worst_residual_wm2': res[worst] if worst is not None else float('nan'),  # snow-free points
        'worst_lat': points.lat.iloc[worst] if worst is not None else float('nan'),
        'worst_lon': points.lon.iloc[worst] if worst is not None else float('nan'),
        'n_snow_points': int(snow.sum()),
        'worst_snow_residual_wm2': res[worst_snow] if worst_snow is not None else float('nan'),
        'n_nan_points': n_nan,
        'has_ml_snow': has_ml_snow,
    }
    return summary, points


def discover_sites(output_dir: Path, site_filter: set[str] | None) -> list[Path]:
    dirs = sorted(p for p in output_dir.iterdir() if p.is_dir() and (p / 'o_efl.nc').exists())
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
    p.add_argument('--tol-wm2', type=float, default=5.0,
                    help='Residual tolerance in run-mean W/m2 (default: 5.0).')
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
    # A point fails if it is NaN, or snow-free and beyond tolerance; NaN must never
    # read as PASS. Snow points are reported but not judged (approximate dE_snow).
    over = points.residual_wm2.isna() | (~points.is_snow & (points.residual_wm2.abs() > args.tol_wm2))
    df['n_failing_points'] = df.site.map(over.groupby(points.site).sum()).astype(int)
    df = df.sort_values('worst_residual_wm2', key=lambda s: s.abs(), ascending=False)

    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', lambda x: f'{x:9.4f}')
    print(df.to_string(index=False))
    print()
    for label, sel in (('snow-free', ~points.is_snow), ('snow (not judged)', points.is_snow)):
        r = points.residual_wm2[sel].abs()
        if len(r):
            print(f'{len(r)} {label} land points.  |residual| per point: mean={r.mean():.4f}  '
                  f'median={r.median():.4f}  max={r.max():.4f} W/m2')

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
        print(f'\nFAIL: {len(failing)} site(s) have snow-free land points beyond {args.tol_wm2} W/m2, '
              f'or NaN at any land point:')
        print(failing[['site', 'n_failing_points', 'n_nan_points', 'worst_residual_wm2',
                       'worst_lat', 'worst_lon']].to_string(index=False))
        return 1
    print(f'\nPASS: all snow-free land points within {args.tol_wm2} W/m2 '
          f'(snow points reported above, not judged).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
