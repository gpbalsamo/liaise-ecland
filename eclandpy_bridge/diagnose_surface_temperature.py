#!/usr/bin/env python3
"""Decompose the eclandpy-vs-Fortran surface-temperature difference on the LIAISE domain.

The dashboards plot eclandpy's AvgSurfT (skin, o_gg) against the control's T2m (o_d2m) because the
port has no 2 m diagnostic, and a skin temperature carries a larger seasonal and diurnal amplitude
than a 2 m temperature. This separates that definitional effect from any real model difference by
putting three series on the same cells and time stamps:

    A  Fortran T2m        (o_d2m)   -- the grey line on the dashboards
    B  Fortran AvgSurfT   (o_gg)    -- the same model, skin
    C  eclandpy AvgSurfT  (o_gg)    -- the blue line

B - A is the definitional offset inside the Fortran itself; C - B is the actual port difference.
Reported as a monthly climatology, a JJA/DJF diurnal cycle, and split by cells with lake cover.

Usage:
  python3 diagnose_surface_temperature.py --years 1988-2024 --eclandpy output_v2 \
      --out /perm/pad/liaise_discharge_compare/surface_temperature_decomposition.json
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


def masks():
    with nc.Dataset(SURFCLIM) as ds:
        land = np.asarray(ds["Mask"][:]) > 0
        clake = np.asarray(ds["CLAKE"][:]).astype(float)
    return land.reshape(-1), (land & (clake > 0.0)).reshape(-1)


def field(path: Path, var: str, times=None):
    with nc.Dataset(path) as ds:
        t = np.asarray(ds["time"][:])
        idx = slice(None) if times is None else np.nonzero(np.isin(t, times))[0]
        a = np.ma.filled(np.asarray(ds[var][idx]), np.nan).astype(np.float64)
        t = t[idx]
    return a.reshape(a.shape[0], -1), t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", default="1988-2024")
    p.add_argument("--eclandpy", default="output_v2")
    p.add_argument("--out", type=Path, default=Path("/perm/pad/liaise_discharge_compare/surface_temperature_decomposition.json"))
    a = p.parse_args()
    y0, _, y1 = a.years.partition("-")
    years = list(range(int(y0), int(y1 or y0) + 1))
    land, lake = masks()
    groups = {"all_land": land, "lake": lake, "other_land": land & ~lake}

    # accumulators: [group][series][month] -> (sum, n); diurnal: [group][series][season][hour]
    acc = {g: {s: np.zeros((12, 2)) for s in "ABC"} for g in groups}
    diu = {g: {s: {"DJF": np.zeros((24, 2)), "JJA": np.zeros((24, 2))} for s in "ABC"} for g in groups}
    per_year = {}

    for y in years:
        f_gg, f_d2m = CONTROL / str(y) / "o_gg.nc", CONTROL / str(y) / "o_d2m.nc"
        e_gg = BRIDGE / a.eclandpy / str(y) / "o_gg.nc"
        if not (f_gg.exists() and f_d2m.exists() and e_gg.exists()):
            continue
        B, t = field(f_gg, "AvgSurfT")
        A, _ = field(f_d2m, "T2m", t)
        C, _ = field(e_gg, "AvgSurfT", t)
        n = min(len(A), len(B), len(C))
        A, B, C, t = A[:n], B[:n], C[:n], t[:n]
        hours = ((t / 3600.0).astype(int)) % 24
        doy = (t / 86400.0).astype(int)
        month = np.array([(np.datetime64(f"{y}-01-01") + int(d)).astype("datetime64[M]").astype(int) % 12 for d in doy])
        season = np.where(np.isin(month, [11, 0, 1]), "DJF", np.where(np.isin(month, [5, 6, 7]), "JJA", ""))
        rec = {}
        for g, sel in groups.items():
            for name, arr in (("A", A), ("B", B), ("C", C)):
                m = np.nanmean(arr[:, sel], axis=1)  # domain mean per record
                for k in range(12):
                    s = month == k
                    if s.any():
                        acc[g][name][k] += (np.nansum(m[s]), s.sum())
                for ss in ("DJF", "JJA"):
                    for h in range(24):
                        s = (season == ss) & (hours == h)
                        if s.any():
                            diu[g][name][ss][h] += (np.nansum(m[s]), s.sum())
                rec[name] = float(np.nanmean(m))
        per_year[str(y)] = {k: round(v - 273.15, 3) for k, v in rec.items()}
        print(f"{y}: T2m {rec['A']-273.15:6.2f}  Fortran skin {rec['B']-273.15:6.2f}  eclandpy skin {rec['C']-273.15:6.2f}"
              f"  | skin-2m {rec['B']-rec['A']:+.2f}  eclandpy-Fortran skin {rec['C']-rec['B']:+.3f}", flush=True)

    def clim(d):
        return {s: [round(float(d[s][k, 0] / d[s][k, 1] - 273.15), 3) if d[s][k, 1] else None for k in range(12)] for s in "ABC"}

    out = {"years": per_year, "monthly": {g: clim(acc[g]) for g in groups},
           "diurnal": {g: {s: {ss: [round(float(diu[g][s][ss][h, 0] / diu[g][s][ss][h, 1] - 273.15), 3)
                                    if diu[g][s][ss][h, 1] else None for h in range(24)] for ss in ("DJF", "JJA")}
                           for s in "ABC"} for g in groups},
           "legend": {"A": "Fortran T2m (o_d2m)", "B": "Fortran AvgSurfT (o_gg)", "C": "eclandpy AvgSurfT (o_gg)"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    m = out["monthly"]["all_land"]
    print(f"\nmonthly climatology, all land ({len(per_year)} years), degC")
    print(f"{'':22s} " + " ".join(f"{x:>6s}" for x in "JFMAMJJASOND"))
    for s, lab in (("A", "Fortran T2m"), ("B", "Fortran AvgSurfT"), ("C", "eclandpy AvgSurfT")):
        print(f"{lab:22s} " + " ".join(f"{v:6.2f}" for v in m[s]))
    print(f"{'skin - 2m (Fortran)':22s} " + " ".join(f"{b-x:+6.2f}" for b, x in zip(m["B"], m["A"])))
    print(f"{'eclandpy - Fortran skin':22s} " + " ".join(f"{c-b:+6.2f}" for c, b in zip(m["C"], m["B"])))
    amp = lambda v: max(v) - min(v)
    print(f"\nseasonal amplitude (max-min month): Fortran T2m {amp(m['A']):.2f} K, "
          f"Fortran skin {amp(m['B']):.2f} K, eclandpy skin {amp(m['C']):.2f} K")
    for ss in ("DJF", "JJA"):
        d = out["diurnal"]["all_land"]
        print(f"{ss} diurnal amplitude: Fortran T2m {amp(d['A'][ss]):.2f} K, Fortran skin {amp(d['B'][ss]):.2f} K, "
              f"eclandpy skin {amp(d['C'][ss]):.2f} K | eclandpy-Fortran skin at min {min(range(24), key=lambda h: d['B'][ss][h])}h "
              f"{d['C'][ss][min(range(24), key=lambda h: d['B'][ss][h])] - d['B'][ss][min(range(24), key=lambda h: d['B'][ss][h])]:+.2f} K, "
              f"at max {max(range(24), key=lambda h: d['B'][ss][h])}h "
              f"{d['C'][ss][max(range(24), key=lambda h: d['B'][ss][h])] - d['B'][ss][max(range(24), key=lambda h: d['B'][ss][h])]:+.2f} K")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
