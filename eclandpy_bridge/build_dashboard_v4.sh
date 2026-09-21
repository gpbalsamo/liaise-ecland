#!/bin/bash
# Build the v4 eclandpy LIAISE dashboard from output_v4 (the 10-tile, LEURBAN=T control run).
#
# v4 is the first control run with the urban tile active, i.e. the configuration ecLand itself
# runs by default (LEURBAN=.TRUE., 10 tiles). v1-v3 were 9-tile runs; they are left untouched so
# the dashboards stay comparable.
#
# Stage A (this script, run as a job that depends on the 37-year run):
#   land/energy diagnostics -> dashboard_eclandpy_v4/index.html, then submit the CaMa-Flood-GPU
#   chain and queue Stage B behind it.
# Stage B: rebuild the same index.html with the discharge and GRDC-skill panels added.
#
# Usage: bash build_dashboard_v4.sh   (normally submitted with --dependency=afterok:<liaise job>)

set -eu
set -o pipefail

B=/perm/pad/liaise-ecland/eclandpy_bridge
D=/perm/pad/liaise_discharge_compare
E=/etc/ecmwf/nfs/dh2_perm_a/pad/eclandpy
YEARS=1988-2024

cd "$B"
module load python3/3.11.10-01
source "$E/.venv-physics/bin/activate"

echo "== stage A: land diagnostics from output_v4 =="
python3 extract_eclandpy_diagnostics.py --output-root "$B/output_v4" --years "$YEARS" \
        --out "$D/eclandpy_run_diagnostics_v4.json"
python3 diagnose_surface_temperature.py --eclandpy output_v4 --years "$YEARS" \
        --out "$D/surface_temperature_decomposition_v4.json"
python3 compare_eclandpy_versions.py --v1 output_v3 --v2 output_v4 --years "$YEARS" \
        --out "$D/v3_v4_vs_fortran.json"

NOTE='v4 is the first control run with the urban tile active (LEURBAN=T, 10 tiles), matching ecLand default.'
build() {  # $@ = extra args
  python3 build_eclandpy_dashboard.py \
    --eclandpy "$D/eclandpy_run_diagnostics_v4.json" \
    --control  "$D/control_run_diagnostics.json" \
    --out      "$B/dashboard_eclandpy_v4/index.html" \
    --label v4 --note "$NOTE" "$@"
}

mkdir -p "$B/dashboard_eclandpy_v4"
build
echo "== stage A done: $B/dashboard_eclandpy_v4/index.html (land/energy panels) =="

echo "== submitting the CaMa-Flood-GPU chain for the discharge + GRDC panels =="
bash submit_cmfgpu_eclandpy_chain.sh 1988 2024 _v4 output_v4
sleep 10
CHAIN=$(squeue -u "$USER" -h -n cmfgpu_eclpy -o %i | tail -1)
if [ -z "$CHAIN" ]; then
  echo "WARNING: could not find the cmfgpu_eclpy job; rerun stage B by hand once it finishes."
  exit 0
fi
echo "   chain job = $CHAIN"

sbatch --job-name="dash_v4_b" --output="$B/logs/dashboard_v4_stageB.out" \
       --error="$B/logs/dashboard_v4_stageB.out" \
       --account=ecrdmocp --partition=par --time=1:00:00 --nodes=1 --ntasks=1 --cpus-per-task=1 \
       --dependency="afterok:$CHAIN" --wrap="
  set -eu
  module load python3/3.11.10-01
  cd '$B'; source '$E/.venv-physics/bin/activate'
  python3 build_eclandpy_dashboard.py \
    --eclandpy '$D/eclandpy_run_diagnostics_v4.json' \
    --control  '$D/control_run_diagnostics.json' \
    --grdc     '$D/skill_benchmark_results_eclandpy_v4.json' \
    --discharge '$B/cmfgpu_out_gpu_v4/eclandpy_liaise_*_discharge_daily.nc' \
    --out '$B/dashboard_eclandpy_v4/index.html' --label v4 --note '$NOTE'
  echo 'stage B done: dashboard_eclandpy_v4/index.html with discharge + GRDC panels'
"
echo "== stage B queued behind $CHAIN =="
