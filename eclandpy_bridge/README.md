# eclandpy on the LIAISE domain — the all-Python ecLand → CaMa-Flood chain

eclandpy (Python/GT4Py port of ecLand, cy50r1 physics) for the land surface, CaMa-Flood-GPU
for river routing, 37 years (1988–2024) on the LIAISE grid (16×23, Ebro basin), compared
throughout with this repo's Fortran ecLand CY50R1 control run. Dashboard:
[sites.ecmwf.int/pad/liaise/eclandpy/](https://sites.ecmwf.int/pad/liaise/eclandpy/), beside
the control at [/pad/liaise/control/](https://sites.ecmwf.int/pad/liaise/control/).

Everything generic lives in two packages; this directory is only the LIAISE configuration of
them, the way `namelist/` configures the Fortran run:

- **eclandpy** (`github.com/gpbalsamo/eclandpy`) — the land model, and `eclandpy.cmfgpu`, the
  coupling layer to CaMa-Flood-GPU (runoff hand-over, weights, chained driver, discharge).
- **CaMa-Flood-GPU** (`github.com/Kshy0/CaMa-Flood-GPU`, Kang, Yin & Yamazaki 2026) — the
  routing model, unmodified.

## Reproduce from a fresh clone

Two environments are unavoidable: eclandpy's land physics needs gt4py 1.1.7 / numpy<2
(`.venv-physics`, see eclandpy's README "Physics core environment"); CaMa-Flood-GPU needs
Python 3.13 + torch/triton. The coupling modules run in either.

```bash
# 0. inputs this repo already carries (Git LFS): init_clim/data/{surfclim,soilinit},
#    cama_flood/data/{inpmat,ncdata}.nc; forcing via forcing/ (see the top-level README)
git lfs pull

# 1. LIAISE inputs in eclandpy's consolidated-file convention, one file per year
python3 prepare_liaise_for_eclandpy.py --year-start 1988 --year-end 2024    # -> data/

# 2. the 37-year land run, year-by-year restart chain, on one A100  (~6.5 h; CPU: ~19 h)
bash submit_eclandpy_liaise_control_gpu.sh            # or submit_eclandpy_liaise_control.sh
#    resumable: `... 1998 2024` continues from restart_gpu/eclandpy_restart_1997.pkl

# 3. routing: -(Qs+Qsb) -> chained CaMa-Flood-GPU -> daily discharge -> GRDC gauge skill (~9 min)
bash submit_cmfgpu_eclandpy_chain.sh 1988 2024

# 4. dashboard
python3 extract_eclandpy_diagnostics.py --years 1988-2024
python3 build_eclandpy_dashboard.py                    # -> dashboard_eclandpy/index.html
```

Step 3's CaMa-Flood-GPU inputs (`/perm/pad/CaMa-Flood-GPU-run/inp/liaise/`) are built once
from this repo's own validated weights and CaMa-Flood-GPU's global map package:
`eclandpy-cmfgpu-mapping --inpmat cama_flood/data/inpmat.nc --ncdata cama_flood/data/ncdata.nc`
and `eclandpy-cmfgpu-subset --parameters <global parameters.nc> --ncdata cama_flood/data/ncdata.nc`
(`CLAUDE.md`, "CaMa-Flood-GPU coupling prep", records how and what was validated).

## What is in here

| file | role |
|---|---|
| `prepare_liaise_for_eclandpy.py` | surfclim/soilinit/WFDE5 → `surfclim_/surfinit_/met_2dHT_LIAISE_<y>-<y>.nc` (`forcing_type="2d"`; time axis in seconds since run start; forcing needs the extra endpoint record) |
| `run_eclandpy_liaise_control.py` | year-by-year chain with a restart (ecland_porting has none): pickles `OfflineState.host()` + carbon accumulators, generic over the dataclass fields; `--smoke-steps` for loop tests |
| `submit_eclandpy_liaise_control{,_gpu}.sh` | SLURM: CPU (`gt:cpu_kfirst`) and A100 (`gt:gpu`); the GPU one documents every environment line it needs |
| `submit_cmfgpu_eclandpy_chain.sh` | LIAISE configuration of `eclandpy.cmfgpu` + `cama_flood/skill_benchmark_fortran_vs_gpu.py` |
| `extract_eclandpy_diagnostics.py`, `build_eclandpy_dashboard.py` | control-run-shaped diagnostics JSON; self-contained dashboard drawn against the control |
| `test_*.sh` | the toolchain recipes (GPU backend, recorders + restart on GPU, timing, CaMa-Flood-GPU build) as executable record |

Run artefacts (`data/`, `output*/`, `restart*/`, `cmfgpu_*/`, `dashboard_eclandpy/`, `logs/`)
are gitignored, as `run/output` and `run/restart` are.

Reproducibility check (2026-09-15): step 3 re-run from scratch through the packaged
`eclandpy.cmfgpu` (`bash submit_cmfgpu_eclandpy_chain.sh 1988 2024 _repro`) gives daily discharge
bit-identical to the original run in all 37 years (max |Δ| = 0) and the same GRDC scores.

`docs/build_eclandpy_docs.py` (top-level `docs/`) regenerates the Word report and slide deck
from the diagnostics JSONs and `docs/figures/`.

## Results (37 years, domain means over the 235 active land points)

Two eclandpy chains exist. **v1** (Sept 2026, GPU) is the run the report and slides describe;
**v2** (2026-09-20, CPU, `output_v2/`) repeats it on `ecland_porting` `cy50r1` after N. Wedi's
FLake lake tile and KSTEP==0 flux estimate were merged. The Fortran control was itself re-run in
September as a proper restart chain (it had been 37 independent one-year cold starts, each with a
spurious start-up runoff pulse), and `control_run_diagnostics.json` regenerated on 17 September —
so the control column below is **not** the one the published v1 dashboard shows.

| | eclandpy v1 | eclandpy v2 | Fortran control | bias v1 | bias v2 |
|---|---|---|---|---|---|
| precipitation, mm/yr | 812.7 | 812.7 | 812.7 | 0 | 0 |
| evapotranspiration | −654.5 | −658.1 | −662.2 | +7.7 | **+4.2** |
| total runoff −(Qs+Qsb) | 160.8 | 161.8 | 157.5 | +3.3 (+2.1 %) | +4.3 (+2.7 %) |
| surface T (AvgSurfT vs T2m), °C | 12.61 | 12.62 | 12.57 | +0.04 | +0.05 |
| root-zone moisture, kg/m² | 410.8 | 411.9 | 405.9 | +4.9 | +6.0 |

Annual correlation with the control is 0.999 for precipitation, evaporation and runoff, 0.989 for
surface temperature. The 21 % runoff gap and the 34.6 kg/m² root-zone gap on the v1 dashboard were
artefacts of the stale control and disappear for both versions.

`compare_eclandpy_versions.py` compares two chains with the same control at every model step and
cell (matching eclandpy's half-hourly records to the control's hourly ones on time values), split
by cells that carry lake cover. Over the 37 years:

| RMSE vs control | lake cells (220) | other land (15) | all land (235) |
|---|---|---|---|
| AvgSurfT, K | 0.344 → 0.331 (−3.9 %) | 0.2990 → 0.2990 (0.00 %) | 0.341 → 0.329 (−3.7 %) |
| SoilTemp, K | 0.208 → 0.195 (−6.1 %) | unchanged | 0.208 → 0.196 (−5.7 %) |
| SWE, kg/m² | 0.514 → 0.327 (−36.3 %) | unchanged | 0.498 → 0.318 (−36.3 %) |
| SoilMoist, kg/m² | 11.14 → 11.21 (+0.6 %) | unchanged | 11.07 → 11.14 (+0.6 %) |

The cells without lake cover are bit-identical between v1 and v2, so every difference is the lake
tile. Soil moisture is the one term that moves the wrong way, and its bias changes sign between
groups (lake cells +0.79 → +1.23 kg/m², other land −0.60) — the deep-layer divergence noted in the
restart check below is the open question.

Restart check (eclandpy state at 1998-01-01 after ten chained years vs the Fortran run's own
`restart_in.nc`): identical at 1988 t=0; soil temperature within 0.1 K and soil moisture within
1–2 % in all four layers after ten years; one clean timestep across the 1997→1998 boundary. The
deep layer (1.9 m) carries a regionally coherent divergence — eclandpy drier on the Duero
plateau, wetter on the coasts, 35 of 235 cells by >100 kg/m² — that domain means hide.

Discharge at the six GRDC gauges (25 station-years, Guadalope excluded as a regulated river):
median KGE −0.135 (v1) and −0.134 (v2) against −0.155 for the Fortran chain, median r 0.451 for
both versions vs 0.342, eclandpy ahead at 15 (v1) / 14 (v2) of 25 station-years, and drier
(PBIAS −42 % vs −36 %). The lake tile moves routed discharge by ~1 mm/yr of runoff, so the gauge
skill is unchanged between v1 and v2.

Dashboards: [/pad/liaise/eclandpy/v1/](https://sites.ecmwf.int/pad/liaise/eclandpy/v1/) (as
published 15 September, against the stale control) and
[/pad/liaise/eclandpy/v2/](https://sites.ecmwf.int/pad/liaise/eclandpy/v2/) (this run, against the
corrected control); the landing page `/pad/liaise/eclandpy/` still serves the original v1 page.

Performance on the 368-column grid, per 30-min step (17,520 steps/year): Fortran 8.8 ms
(2.56 min/year, 4 OpenMP threads); eclandpy **24 ms on one CPU core** (`gt:cpu_kfirst`,
≈7 min/year) and **32 ms on one A100** (`gt:gpu`; 37 ms with the three recorders = the
10.8 min/year the 37-year run measured). The CPU run archived in `output/` took 30.5 min/year
(104 ms/step) because it ran on ecland-porting *before* C. Kühnlein's main merge (`29e6152`,
2026-09-14: `run_frozen`, storage cache), which made the CPU path 4× faster; an earlier
"2.9× GPU speed-up" compared across those two code versions and was wrong. On this domain one
CPU core beats the A100: 368 columns cannot fill a GPU (per-step cost is launch latency;
CaMa-Flood-GPU's own benchmarks run at 17,675×N columns), so the port pays off at scale, not
here. Timing scripts: `profile_liaise_step.py` (250-step blocks) and
`profile_liaise_recorders.py` (cProfile, recorders, write cost) — run them with the eclandpy
tree as the working directory (gt4py roots its stencil cache on the cwd) and the environment
of the submit scripts. The three `nfrpos=1` recorders cost ~3 ms/step on the A100; a variant
keeping the samples on the device and transferring them in 256-record blocks was tried
(2026-09-15, bit-identical files) and gained nothing, so the cost is not the device→host
copies and the host recorders stay. Output files are time-chunked since ecland-porting
`c3f6846` (fork branch `cy50r1`; same fix as ecLand's NCHUNKTIME): unchanged values, ~5×
smaller point-site files, 45× faster time-series reads.

## Two known gaps in the Python coupling (both off in the reference configuration)

`LWEVAP` — the Fortran hands CaMa-Flood a third field, lake-tile potential evaporation
(`cnt41s.F90`: `-D1STIEVAPU2(:,9,:,:)`), extracted from floodplain storage; CaMa-Flood-GPU has
no evaporation sink. `LROSPLIT` — surface and sub-surface runoff are passed separately; the
Python chain combines them, equivalent while `LGDWDLY` is off. `namelist/input_cmf` has
`LWEVAP=false`, `LGDWDLY=false`.

Lake tile: ecland-porting `cy50r1` assigns the FLake tile `PFRTI(:,9)` in `surfbc` since
2026-09-19 (N. Wedi's PR #1, merged as `c8385de`, plus PR #2's KSTEP==0 flux estimate and the
`surfbc_class` twin fix `93728d5`). The **v1** chain predates it; **v2** is the 37-year rerun with
it, quantified in Results above. PLUMBER2 sites have no lake cover and are bit-identical.
