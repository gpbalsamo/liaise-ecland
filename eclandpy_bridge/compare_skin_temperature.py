#!/usr/bin/env python3
"""Skin-vs-skin surface temperature: eclandpy AvgSurfT against the Fortran control's AvgSurfT.

The dashboards have always plotted eclandpy's AvgSurfT (skin, o_gg) against the control's T2m
(o_d2m), because run/extract_control_diagnostics.py was written for the Fortran-only dashboard
and only ever opens o_d2m.nc. The Fortran writes its own skin temperature in o_gg.nc, so the
like-for-like comparison was always available; this does it.

Two inconsistencies in the dashboard pairing, both removed here:
  1. different variable -- skin against 2 m air, which differ by ~1 K seasonally even inside the
     same model (see diagnose_surface_temperature.py);
  2. different mask -- the control extractor's masked_mean() masks only fill values, while the
     eclandpy extractor averages over surfclim Mask>0. Everything below uses Mask>0.

The Fortran side (series B) is read from the stored decomposition JSON rather than recomputed:
the control's raw netCDF was purged from scratch, and B is a property of the control run, which
has not changed. The eclandpy side is recomputed from the run given by --eclandpy, using the same
mask, the same per-record domain mean and the same monthly accumulation as that decomposition, so
the two sides stay comparable.

Usage:
  python3 compare_skin_temperature.py --eclandpy /ec/res4/.../output_v5 --label v5 \
      --fortran /perm/pad/liaise_discharge_compare/surface_temperature_decomposition_v4.json \
      --out /perm/pad/liaise_discharge_compare/skin_vs_skin_v5.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

SURFCLIM = Path("/perm/pad/liaise-ecland/init_clim/data/surfclim")


def masks():
    with nc.Dataset(SURFCLIM) as ds:
        land = np.asarray(ds["Mask"][:]) > 0
    return land.reshape(-1)


def field(path: Path, var: str):
    with nc.Dataset(path) as ds:
        t = np.asarray(ds["time"][:])
        a = np.ma.filled(np.asarray(ds[var][:]), np.nan).astype(np.float64)
    return a.reshape(a.shape[0], -1), t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eclandpy", type=Path, required=True, help="eclandpy output root (holds <year>/o_gg.nc)")
    p.add_argument("--fortran", type=Path,
                   default=Path("/perm/pad/liaise_discharge_compare/surface_temperature_decomposition_v4.json"),
                   help="decomposition JSON carrying the control's skin temperature as series B")
    p.add_argument("--label", default="v5")
    p.add_argument("--years", default="1988-2024")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    y0, _, y1 = a.years.partition("-")
    years = list(range(int(y0), int(y1 or y0) + 1))

    land = masks()
    fort = json.load(open(a.fortran))
    fort_year = {y: v["B"] for y, v in fort["years"].items()}       # degC, Fortran skin
    fort_month = fort["monthly"]["all_land"]["B"]                   # degC, 12 values

    acc = np.zeros((12, 2))
    per_year = {}
    for y in years:
        f = a.eclandpy / str(y) / "o_gg.nc"
        if not f.exists():
            print(f"{y}: no o_gg.nc -- skipped", flush=True)
            continue
        C, t = field(f, "AvgSurfT")
        m = np.nanmean(C[:, land], axis=1)                          # domain mean per record, K
        doy = (t / 86400.0).astype(int)
        month = np.array([(np.datetime64(f"{y}-01-01") + int(d)).astype("datetime64[M]").astype(int) % 12
                          for d in doy])
        for k in range(12):
            s = month == k
            if s.any():
                acc[k] += (np.nansum(m[s]), s.sum())
        c = float(np.nanmean(m)) - 273.15
        b = fort_year.get(str(y))
        per_year[str(y)] = {"eclandpy_skin": round(c, 3), "fortran_skin": b,
                            "diff": round(c - b, 3) if b is not None else None}
        if b is not None:
            print(f"{y}: eclandpy skin {c:6.2f}  Fortran skin {b:6.2f}  diff {c - b:+.3f} K", flush=True)

    ecl_month = [round(float(acc[k, 0] / acc[k, 1] - 273.15), 3) if acc[k, 1] else None for k in range(12)]
    paired = [v for v in per_year.values() if v["diff"] is not None]
    mean_e = sum(v["eclandpy_skin"] for v in paired) / len(paired)
    mean_f = sum(v["fortran_skin"] for v in paired) / len(paired)

    out = {"label": a.label,
           "note": ("eclandpy AvgSurfT vs Fortran control AvgSurfT, both over surfclim Mask>0. "
                    "The Fortran side is series B of " + a.fortran.name + "; the control run is "
                    "unchanged, its raw netCDF having been purged from scratch."),
           "years": per_year,
           "monthly_all_land": {"eclandpy_skin": ecl_month, "fortran_skin": fort_month,
                                "diff": [round(e - f, 3) if (e is not None and f is not None) else None
                                         for e, f in zip(ecl_month, fort_month)]},
           "summary": {"n_years": len(paired),
                       "eclandpy_skin_mean_c": round(mean_e, 3),
                       "fortran_skin_mean_c": round(mean_f, 3),
                       "diff_k": round(mean_e - mean_f, 4)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    s = out["summary"]
    print(f"\n{s['n_years']} years: eclandpy skin {s['eclandpy_skin_mean_c']:.3f} C, "
          f"Fortran skin {s['fortran_skin_mean_c']:.3f} C, difference {s['diff_k']:+.4f} K")
    print("monthly diff (K): " + " ".join(f"{d:+.2f}" for d in out["monthly_all_land"]["diff"]))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
