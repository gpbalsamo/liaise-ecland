#!/bin/bash
# Build the v5 eclandpy LIAISE dashboard from output_v5.
#
# v5 is the first control run on upstream ecland-porting main (d61221c) after the urban-tile work
# was merged there: it carries Christian's tile-vector refactor (PR #8) and Nils's IFS host
# integration (PR #7) on top of the 10-tile LEURBAN=T configuration v4 introduced. PR #8
# deliberately reproduces ecLand's executed VSURF bare-ground fraction, PFRTI(8)+PFRTI(9), which
# omits the urban term -- so v5 is not expected to be bit-identical to v4.
#
# Stage A (this script, run as a job that depends on the 37-year run):
#   land/energy diagnostics -> dashboard_eclandpy_v5/index.html, then submit the CaMa-Flood-GPU
#   chain and queue Stage B behind it.
# Stage B: rebuild the same index.html with the discharge and GRDC-skill panels added.
#
# Unlike build_dashboard_v4.sh this skips diagnose_surface_temperature.py and
# compare_eclandpy_versions.py: both read raw netCDF from runs that scratch has since purged
# (the Fortran control output and output_v3/output_v4). Neither feeds the dashboard itself.
#
# Usage: bash build_dashboard_v5.sh   (normally submitted with --dependency=afterok:<liaise job>)

set -eu
set -o pipefail

B=/perm/pad/liaise-ecland/eclandpy_bridge
D=/perm/pad/liaise_discharge_compare
E=/etc/ecmwf/nfs/dh2_perm_a/pad/eclandpy
YEARS=1988-2024

cd "$B"
module load python3/3.11.10-01
source "$E/.venv-physics/bin/activate"

echo "== stage A: land diagnostics from output_v5 =="
python3 extract_eclandpy_diagnostics.py --output-root "$B/output_v5" --years "$YEARS" \
        --out "$D/eclandpy_run_diagnostics_v5.json"

NOTE='v5 rebases the port onto upstream ecland-porting main (d61221c): the tile-vector refactor and the IFS host integration, on top of the 10-tile LEURBAN=T configuration introduced in v4.'
build() {  # $@ = extra args
  python3 build_eclandpy_dashboard.py \
    --eclandpy "$D/eclandpy_run_diagnostics_v5.json" \
    --control  "$D/control_run_diagnostics.json" \
    --out      "$B/dashboard_eclandpy_v5/index.html" \
    --label v5 --note "$NOTE" "$@"
}

mkdir -p "$B/dashboard_eclandpy_v5"
build
echo "== stage A done: $B/dashboard_eclandpy_v5/index.html (land/energy panels) =="

echo "== submitting the CaMa-Flood-GPU chain for the discharge + GRDC panels =="
bash submit_cmfgpu_eclandpy_chain.sh 1988 2024 _v5 output_v5
sleep 10
CHAIN=$(squeue -u "$USER" -h -n cmfgpu_eclpy -o %i | tail -1)
if [ -z "$CHAIN" ]; then
  echo "WARNING: could not find the cmfgpu_eclpy job; rerun stage B by hand once it finishes."
  exit 0
fi
echo "   chain job = $CHAIN"

sbatch --job-name="dash_v5_b" --output="$B/logs/dashboard_v5_stageB.out" \
       --error="$B/logs/dashboard_v5_stageB.out" \
       --account=ecrdmocp --partition=par --time=1:00:00 --nodes=1 --ntasks=1 --cpus-per-task=1 \
       --dependency="afterok:$CHAIN" --wrap="
  set -eu
  module load python3/3.11.10-01
  cd '$B'; source '$E/.venv-physics/bin/activate'
  python3 build_eclandpy_dashboard.py \
    --eclandpy '$D/eclandpy_run_diagnostics_v5.json' \
    --control  '$D/control_run_diagnostics.json' \
    --grdc     '$D/skill_benchmark_results_eclandpy_v5.json' \
    --discharge '$B/cmfgpu_out_gpu_v5/eclandpy_liaise_*_discharge_daily.nc' \
    --out '$B/dashboard_eclandpy_v5/index.html' --label v5 --note '$NOTE'
  echo 'stage B done: dashboard_eclandpy_v5/index.html with discharge + GRDC panels'
"
echo "== stage B queued behind $CHAIN =="
