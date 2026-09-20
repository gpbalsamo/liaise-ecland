#!/usr/bin/env python3
"""Compare two eclandpy LIAISE chains against the Fortran control, cell by cell and step by step.

Written for the v1 -> v2 comparison (v2 = ecland-porting cy50r1 with N. Wedi's FLake lake tile
PFRTI(:,9) and the KSTEP==0 flux estimate, merged 2026-09-19): the expected signal is confined to
the cells that carry lake cover, so every statistic is reported for the lake cells, the other land
cells and the whole domain separately.

Time axes differ: the Fortran control writes o_gg hourly (NFRPOS=2), eclandpy half-hourly
(nfrpos=1); records are matched on the `time` values, never on index.

Usage:
  python3 compare_eclandpy_versions.py --years 1988-2024 \
      --v1 output_gpu --v2 output_v2 --out /perm/pad/liaise_discharge_compare/v1_v2_vs_fortran.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

BRIDGE = Path("/perm/pad/liaise-ecland/eclandpy_bridge")
CONTROL = Path("/perm/pad/liaise-ecland/run/output")
SURFCLIM = Path("/perm/pad/liaise-ecland/init_clim/data/surfclim")
VARS = ("AvgSurfT", "SoilTemp", "SoilMoist", "SWE")


def masks() -> tuple[np.ndarray, np.ndarray]:
    """(land, lake) boolean (lat, lon) masks: land = surfclim Mask>0, lake = land & CLAKE>0."""
    with nc.Dataset(SURFCLIM) as ds:
        land = np.asarray(ds["Mask"][:]) > 0
        clake = np.asarray(ds["CLAKE"][:]).astype(float)
    return land, land & (clake > 0.0)


def read(path: Path, var: str, times: np.ndarray | None = None):
    """Variable as (time, cell[, level]) plus its time axis, optionally subset to `times`."""
    with nc.Dataset(path) as ds:
        t = np.asarray(ds["time"][:])
        idx = slice(None) if times is None else np.nonzero(np.isin(t, times))[0]
        a = np.ma.filled(np.asarray(ds[var][idx]), np.nan)
        t = t[idx]
    if a.ndim == 4:  # (time, nlevs, lat, lon) -> use the top layer and the deepest layer
        a = a[:, [0, -1]].reshape(a.shape[0], 2, -1)
    else:
        a = a.reshape(a.shape[0], 1, -1)
    return a, t


def stats(model: np.ndarray, ref: np.ndarray, sel: np.ndarray) -> dict:
    d = (model - ref)[:, :, sel]
    return {
        "rmse": float(np.sqrt(np.nanmean(d**2))),
        "bias": float(np.nanmean(d)),
        "max_abs": float(np.nanmax(np.abs(d))),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", default="1988-2024")
    p.add_argument("--v1", default="output_gpu")
    p.add_argument("--v2", default="output_v2")
    p.add_argument("--out", type=Path, default=Path("/perm/pad/liaise_discharge_compare/v1_v2_vs_fortran.json"))
    a = p.parse_args()
    y0, _, y1 = a.years.partition("-")
    years = list(range(int(y0), int(y1 or y0) + 1))

    land, lake = masks()
    land_f, lake_f = land.reshape(-1), lake.reshape(-1)
    other_f = land_f & ~lake_f
    sels = {"lake": lake_f, "other_land": other_f, "all_land": land_f}
    print(f"cells: land {land_f.sum()}, of which lake-fraction {lake_f.sum()}")

    out: dict = {"years": {}, "cells": {k: int(v.sum()) for k, v in sels.items()},
                 "note": "eclandpy half-hourly o_gg matched to the Fortran control's hourly records on time values"}
    for y in years:
        ref_path, v1_path, v2_path = CONTROL / str(y) / "o_gg.nc", BRIDGE / a.v1 / str(y) / "o_gg.nc", BRIDGE / a.v2 / str(y) / "o_gg.nc"
        if not (ref_path.exists() and v1_path.exists() and v2_path.exists()):
            continue
        rec: dict = {}
        for var in VARS:
            ref, tref = read(ref_path, var)
            m1, _ = read(v1_path, var, tref)
            m2, _ = read(v2_path, var, tref)
            n = min(len(ref), len(m1), len(m2))
            ref, m1, m2 = ref[:n], m1[:n], m2[:n]
            rec[var] = {
                grp: {"v1": stats(m1, ref, sel), "v2": stats(m2, ref, sel),
                      "v2_minus_v1": {"max_abs": float(np.nanmax(np.abs((m2 - m1)[:, :, sel]))),
                                      "mean_abs": float(np.nanmean(np.abs((m2 - m1)[:, :, sel])))}}
                for grp, sel in sels.items()
            }
            rec[var]["n_records"] = int(n)
        out["years"][str(y)] = rec
        s = rec["AvgSurfT"]
        print(f"{y}: AvgSurfT RMSE vs Fortran  lake {s['lake']['v1']['rmse']:.3f} -> {s['lake']['v2']['rmse']:.3f}"
              f" | other {s['other_land']['v1']['rmse']:.3f} -> {s['other_land']['v2']['rmse']:.3f}"
              f" | all {s['all_land']['v1']['rmse']:.3f} -> {s['all_land']['v2']['rmse']:.3f}", flush=True)

    # aggregate: RMSE over all years (root of the mean of squared RMSEs weighted by records)
    agg: dict = {}
    for var in VARS:
        agg[var] = {}
        for grp in sels:
            for v in ("v1", "v2"):
                sq = [(r[var][grp][v]["rmse"] ** 2) * r[var]["n_records"] for r in out["years"].values()]
                nrec = [r[var]["n_records"] for r in out["years"].values()]
                bias = [r[var][grp][v]["bias"] * r[var]["n_records"] for r in out["years"].values()]
                agg[var].setdefault(grp, {})[v] = {
                    "rmse": float(np.sqrt(sum(sq) / sum(nrec))) if nrec else float("nan"),
                    "bias": float(sum(bias) / sum(nrec)) if nrec else float("nan"),
                    "max_abs": float(max(r[var][grp][v]["max_abs"] for r in out["years"].values())) if nrec else float("nan"),
                }
    out["all_years"] = agg
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}  ({len(out['years'])} years)")
    print(f"{'variable':12s} {'group':11s} {'RMSE v1':>9s} {'RMSE v2':>9s} {'change':>9s} {'bias v1':>9s} {'bias v2':>9s}")
    for var in VARS:
        for grp in sels:
            s1, s2 = agg[var][grp]["v1"], agg[var][grp]["v2"]
            ch = 100 * (s2["rmse"] - s1["rmse"]) / s1["rmse"] if s1["rmse"] else float("nan")
            print(f"{var:12s} {grp:11s} {s1['rmse']:9.4f} {s2['rmse']:9.4f} {ch:8.2f}% {s1['bias']:9.4f} {s2['bias']:9.4f}")


if __name__ == "__main__":
    main()
