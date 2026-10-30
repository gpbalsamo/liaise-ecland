"""Tests for run/check_energy_budget.py and run/check_water_budget.py.

Each test writes a tiny fake ecLand output year -- a 2 x 3 grid with one sea
cell holding the 1e20 fill value, 24 hourly steps, float32 like the real
files -- whose budget residual at every land point is known exactly. It then
runs the script from the command line as a user would and checks the exit
code (0 = PASS, 1 = FAIL) and the --out-csv table.

The single-site tests (1 x 1 grid, no sea) pin the original PLUMBER2 use the
scripts were written for, and must give the same results before and after
any change to the gridded handling.

    python3 -m pytest tests/ -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset

RUN = Path(__file__).resolve().parents[1] / 'run'
FILL = 1.0e20
NT, DT = 24, 3600.0
RUN_SECONDS = NT * DT
SECONDS_PER_YEAR = 365.25 * 86400.0
SITE = '1988'

LAT, LON = [41.25, 41.75], [0.25, 0.75, 1.25]
SEA = (0, 2)  # (lat index, lon index) of the sea cell

# Soil column, constant with depth: 4 layers, 2.89 m in total
SOIL_THICK = np.array([0.07, 0.21, 0.72, 1.89])
SOIL_FC, SOIL_SAT = 0.30, 0.45
# Independent copy of ecLand's soil heat capacity (sussoil_mod.F90), J K-1 m-3
SOIL_HEAT_CAP = (1.0 - SOIL_SAT) * 1.6e6 + SOIL_FC * 4.18e6

# Water fluxes for a closed budget, kg m-2 s-1 (out-fluxes negative)
RAIN, EVAP, QS, QSB = 1.0e-4, -3.0e-5, -2.0e-5, -1.0e-5


def _grid(nlat: int, nlon: int, sea: tuple[int, int] | None) -> np.ndarray:
    land = np.ones((nlat, nlon), dtype=bool)
    if sea is not None:
        land[sea] = False
    return land


def _new(path: Path, nlat: int, nlon: int, extra_dims: dict[str, int]) -> Dataset:
    ds = Dataset(path, 'w')
    ds.createDimension('time', None)
    ds.createDimension('lat', nlat)
    ds.createDimension('lon', nlon)
    for name, size in extra_dims.items():
        ds.createDimension(name, size)
    ds.createVariable('time', 'f8', ('time',))[:] = (np.arange(NT) + 1) * DT
    ds.createVariable('lat', 'f4', ('lat',))[:] = LAT[:nlat]
    ds.createVariable('lon', 'f4', ('lon',))[:] = LON[:nlon]
    return ds


def _put(ds: Dataset, name: str, dims: tuple[str, ...], value: np.ndarray, land: np.ndarray) -> None:
    """Write a float32 field with FILL at sea points, as ecLand does."""
    var = ds.createVariable(name, 'f4', dims, fill_value=FILL)
    data = np.broadcast_to(value, var.shape).astype('f8').copy()
    data[..., ~land] = FILL
    var[:] = data


def write_energy_year(root: Path, residual: np.ndarray, sea=SEA, soil_warming=None,
                      nan_at=None, snow_at=None) -> Path:
    """Fake o_efl/o_fix/o_vty/o_gg whose closure residual is `residual` (W m-2, per point).

    With soil_warming (K, per point) the soil column warms by that much over
    the run and the fluxes deliver exactly the matching energy, so the correct
    residual is still `residual`. With snow_at, that point carries a constant
    snow pack (no change in snow energy), which makes it a snow point.
    """
    nlat, nlon = residual.shape
    land = _grid(nlat, nlon, sea)
    d = root / SITE
    d.mkdir(parents=True)
    warming = np.zeros_like(residual) if soil_warming is None else soil_warming
    storage_wm2 = SOIL_HEAT_CAP * SOIL_THICK.sum() * warming / RUN_SECONDS
    ground_wm2 = residual + storage_wm2  # energy into the ground, W m-2

    with _new(d / 'o_efl.nc', nlat, nlon, {}) as ds:
        three = ('time', 'lat', 'lon')
        _put(ds, 'LWnet', three, -50.0, land)
        _put(ds, 'Qh', three, -20.0, land)
        _put(ds, 'Qle', three, -30.0, land)
        # SWnet closes the balance: SWnet + LWnet + Qh + Qle = ground_wm2
        sw = np.broadcast_to(100.0 + ground_wm2, (NT, nlat, nlon)).copy()
        if nan_at is not None:
            sw[(0,) + nan_at] = np.nan
        _put(ds, 'SWnet', three, sw, land)

    with _new(d / 'o_fix.nc', nlat, nlon, {'nlevs': 4}) as ds:
        lev = ('nlevs', 'lat', 'lon')
        _put(ds, 'SoilThick', lev, SOIL_THICK[:, None, None], land)
        _put(ds, 'SoilFC', lev, SOIL_FC, land)
        _put(ds, 'SoilSat', lev, SOIL_SAT, land)

    with _new(d / 'o_vty.nc', nlat, nlon, {'vtype': 2}) as ds:
        _put(ds, 'vtfr', ('time', 'vtype', 'lat', 'lon'), np.array([0.3, 0.4])[:, None, None], land)

    with _new(d / 'o_gg.nc', nlat, nlon, {'nlevs': 4, 'nlevsn': 5}) as ds:
        temp = np.full((NT, 4, nlat, nlon), 285.0)
        temp[-1] += warming
        _put(ds, 'SoilTemp', ('time', 'nlevs', 'lat', 'lon'), temp, land)
        snow = ('time', 'nlevsn', 'lat', 'lon')
        swe = np.zeros((NT, 5, nlat, nlon))
        if snow_at is not None:
            swe[(slice(None), 0) + snow_at] = 50.0
        _put(ds, 'SWEML', snow, swe, land)
        _put(ds, 'slwML', snow, 0.0, land)
        _put(ds, 'SnowTML', snow, 273.15, land)
    return root


def write_water_year(root: Path, residual_mm_yr: np.ndarray, sea=SEA, nan_at=None,
                     lake_evap_mm_yr=None) -> Path:
    """Fake o_wat whose water-balance residual is `residual_mm_yr` (per point).

    With lake_evap_mm_yr, that much open-water evaporation is added to Evap
    and written as EWater in o_eva.nc, without touching any land store --
    as ecLand does for its unlimited lake/sea reservoir.
    """
    nlat, nlon = residual_mm_yr.shape
    land = _grid(nlat, nlon, sea)
    d = root / SITE
    d.mkdir(parents=True)
    years = RUN_SECONDS / SECONDS_PER_YEAR
    # Storage absorbs the net inflow, minus whatever residual is wanted
    dsm_per_step = (RAIN + EVAP + QS + QSB) * DT - residual_mm_yr * years / NT
    ewater = np.zeros((nlat, nlon)) if lake_evap_mm_yr is None else -lake_evap_mm_yr / SECONDS_PER_YEAR

    if lake_evap_mm_yr is not None:
        with _new(d / 'o_eva.nc', nlat, nlon, {}) as ds:
            _put(ds, 'EWater', ('time', 'lat', 'lon'), ewater, land)

    with _new(d / 'o_wat.nc', nlat, nlon, {}) as ds:
        three = ('time', 'lat', 'lon')
        rain = np.full((NT, nlat, nlon), RAIN)
        if nan_at is not None:
            rain[(0,) + nan_at] = np.nan
        _put(ds, 'Rainf', three, rain, land)
        _put(ds, 'Snowf', three, 0.0, land)
        _put(ds, 'Evap', three, EVAP + ewater, land)
        _put(ds, 'Qs', three, QS, land)
        _put(ds, 'Qsb', three, QSB, land)
        _put(ds, 'DelSoilMoist', three, dsm_per_step, land)
        _put(ds, 'DelSWE', three, 0.0, land)
        _put(ds, 'DelIntercept', three, 0.0, land)
    return root


def run_script(name: str, output_dir: Path, tmp_path: Path) -> tuple[int, str, pd.DataFrame | None]:
    csv = tmp_path / 'out.csv'
    proc = subprocess.run([sys.executable, str(RUN / name), '--output-dir', str(output_dir),
                           '--out-csv', str(csv)], capture_output=True, text=True)
    df = pd.read_csv(csv) if csv.exists() else None
    return proc.returncode, proc.stdout + proc.stderr, df


def site_row(df: pd.DataFrame | None) -> pd.Series:
    assert df is not None, 'script wrote no CSV'
    return df.loc[df.site.astype(str) == SITE].iloc[0]


# -- energy ------------------------------------------------------------------

ENERGY = 'check_energy_budget.py'


def test_energy_ignores_sea_cell(tmp_path):
    residual = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 4.5]])  # [0, 2] is sea
    code, out, df = run_script(ENERGY, write_energy_year(tmp_path / 'out', residual), tmp_path)
    row = site_row(df)
    assert row.residual_wm2 == pytest.approx(2.9, abs=1e-3), out  # mean of the 5 land points
    assert row.n_points == 5
    assert row.worst_residual_wm2 == pytest.approx(4.5, abs=1e-3)
    assert (row.worst_lat, row.worst_lon) == (LAT[1], LON[2])
    assert code == 0, out


def test_energy_soil_warming_is_subtracted(tmp_path):
    # Fluxes deliver exactly the energy the soil stores: the budget is closed.
    residual = np.zeros((2, 3))
    warming = np.array([[0.01, 0.02, 0.0], [0.03, 0.04, 0.05]])
    code, out, df = run_script(ENERGY, write_energy_year(tmp_path / 'out', residual,
                                                         soil_warming=warming), tmp_path)
    row = site_row(df)
    assert row.residual_wm2 == pytest.approx(0.0, abs=1e-3), out
    assert abs(row.worst_residual_wm2) < 1e-3
    assert code == 0, out


def test_energy_flags_real_imbalance(tmp_path):
    residual = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 10.0]])
    code, out, df = run_script(ENERGY, write_energy_year(tmp_path / 'out', residual), tmp_path)
    row = site_row(df)
    assert row.worst_residual_wm2 == pytest.approx(10.0, abs=1e-3), out
    assert (row.worst_lat, row.worst_lon) == (LAT[1], LON[2])
    assert code == 1, out
    assert 'FAIL' in out


def test_energy_snow_point_reported_not_judged(tmp_path):
    # The snow-energy formula is approximate, so a snow point above tolerance
    # is reported separately but does not fail the check.
    residual = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 8.0]])
    code, out, df = run_script(ENERGY, write_energy_year(tmp_path / 'out', residual,
                                                         snow_at=(1, 2)), tmp_path)
    row = site_row(df)
    assert row.n_points == 5 and row.n_snow_points == 1, out
    assert row.worst_residual_wm2 == pytest.approx(4.0, abs=1e-3)  # worst snow-free point
    assert row.worst_snow_residual_wm2 == pytest.approx(8.0, abs=1e-3)
    assert code == 0, out


def test_energy_nan_at_land_point_fails(tmp_path):
    residual = np.ones((2, 3))
    code, out, _ = run_script(ENERGY, write_energy_year(tmp_path / 'out', residual,
                                                        nan_at=(1, 0)), tmp_path)
    assert code == 1, out
    assert 'FAIL' in out


def test_energy_single_site_unchanged(tmp_path):
    residual = np.array([[1.5]])
    code, out, df = run_script(ENERGY, write_energy_year(tmp_path / 'out', residual, sea=None),
                               tmp_path)
    row = site_row(df)
    assert row.residual_wm2 == pytest.approx(1.5, abs=1e-3), out
    assert row.rnet_mean_wm2 == pytest.approx(51.5, abs=1e-3)
    assert row.dE_soil_MJ_m2 == pytest.approx(0.0, abs=1e-6)
    assert code == 0, out


# -- water -------------------------------------------------------------------

WATER = 'check_water_budget.py'
PRECIP_MM_YR = RAIN * SECONDS_PER_YEAR  # 3155.76


def test_water_ignores_sea_cell(tmp_path):
    code, out, df = run_script(WATER, write_water_year(tmp_path / 'out', np.zeros((2, 3))), tmp_path)
    row = site_row(df)
    assert row.precip_mm_yr == pytest.approx(PRECIP_MM_YR, rel=1e-4), out
    assert row.residual_mm_yr == pytest.approx(0.0, abs=0.5)
    assert row.n_points == 5
    assert code == 0, out


def test_water_flags_real_imbalance(tmp_path):
    residual = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]])  # 100 mm/yr = 3% of P
    code, out, df = run_script(WATER, write_water_year(tmp_path / 'out', residual), tmp_path)
    row = site_row(df)
    assert row.precip_mm_yr == pytest.approx(PRECIP_MM_YR, rel=1e-4), out
    assert row.worst_residual_mm_yr == pytest.approx(100.0, abs=0.5)
    assert (row.worst_lat, row.worst_lon) == (LAT[1], LON[2])
    assert code == 1, out
    assert 'FAIL' in out


def test_water_lake_evaporation_excluded(tmp_path):
    # Open-water evaporation (EWater) leaves Evap without draining any land
    # store; the balance must still close.
    lake = np.array([[0.0, 0.0, 0.0], [0.0, 50.0, 0.0]])  # 50 mm/yr from a lake at [1, 1]
    code, out, df = run_script(WATER, write_water_year(tmp_path / 'out', np.zeros((2, 3)),
                                                       lake_evap_mm_yr=lake), tmp_path)
    row = site_row(df)
    assert row.residual_mm_yr == pytest.approx(0.0, abs=0.5), out
    assert abs(row.worst_residual_mm_yr) < 0.5
    assert code == 0, out


def test_water_nan_at_land_point_fails(tmp_path):
    code, out, _ = run_script(WATER, write_water_year(tmp_path / 'out', np.zeros((2, 3)),
                                                      nan_at=(1, 0)), tmp_path)
    assert code == 1, out
    assert 'FAIL' in out


def test_water_single_site_unchanged(tmp_path):
    code, out, df = run_script(WATER, write_water_year(tmp_path / 'out', np.zeros((1, 1)), sea=None),
                               tmp_path)
    row = site_row(df)
    assert row.precip_mm_yr == pytest.approx(PRECIP_MM_YR, rel=1e-4), out
    assert row.residual_mm_yr == pytest.approx(0.0, abs=0.5)
    assert code == 0, out
