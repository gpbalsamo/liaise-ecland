#!/bin/bash
# Resume the v5 37-year eclandpy LIAISE control run after the PERM quota failure.
#
# The first attempt (job 33526181) died 6.5 min into 1999 having written 1988-1998. Its log was
# empty -- stdout block-buffered and lost, no traceback on stderr -- because nothing had gone
# wrong in the model: $PERM was full (9.99T of 10T) and the write failed. Output and restarts
# therefore live on SCRATCH now, and this script lives here too because PERM will not take even
# a symlink. run_eclandpy_liaise_control.py picks up eclandpy_restart_<year-1>.pkl by itself, so
# the eleven completed years are not redone. python3 -u so a repeat failure leaves its error.
#
# Usage: bash submit_v5_resume.sh [year_start] [year_end]

set -eu
set -o pipefail

ECLANDPY_ROOT="/etc/ecmwf/nfs/dh2_perm_a/pad/eclandpy"
BRIDGE_ROOT="/perm/pad/liaise-ecland/eclandpy_bridge"
SCR="/ec/res4/scratch/pad/liaise-ecland/eclandpy_v5"
Y0="${1:-1999}"
Y1="${2:-2024}"

sbatch \
  --job-name="eclpy_v5r" \
  --output="${SCR}/logs/control_run_v5_resume_${Y0}.out" \
  --error="${SCR}/logs/control_run_v5_resume_${Y0}.out" \
  --account=ecrdmocp \
  --partition=par \
  --time=12:00:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --wrap="
    set -eu
    module load python3/3.11.10-01
    cd '${ECLANDPY_ROOT}'
    source .venv-physics/bin/activate
    export PYTHONPATH=/perm/pad/eclandpy/deps/sv-overlay/src
    export CXX=icpx CC=icx
    export GT4PY_EXTRA_COMPILE_ARGS='-fbracket-depth=65536 -fconstexpr-steps=2000000000 -fconstexpr-depth=4096'
    export ECLAND_BACKEND=gt:cpu_kfirst
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

    python3 -u '${BRIDGE_ROOT}/run_eclandpy_liaise_control.py' \
      --year-start ${Y0} --year-end ${Y1} --with-fluxes \
      --out-root '${SCR}/output_v5' --restart-root '${SCR}/restart_v5'
  "

echo "Submitted. Check with: squeue -u \$USER -n eclpy_v5r"
