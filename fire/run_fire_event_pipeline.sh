#!/usr/bin/env bash
set -euo pipefail

# Reproduce the LEFIRE fire-event dryness validation end to end: fetch fire records,
# build the domain catalogue, extract a daily-moisture cache from a finished LEFIRE run,
# draw every event's figure, and score the dryness signal against a trend-controlled
# baseline. See CLAUDE.md, "Fire danger (LEFIRE)".
#
# Requires a finished LEFIRE run (namelist/input_fire, LWRFIRE=true so o_fire.nc is
# written) -- point RUN_ROOT at its RUN_ROOT. This step does not run ecLand itself.
#
# Usage
#   RUN_ROOT=/path/to/run_1988_2024 fire/run_fire_event_pipeline.sh
#
# Re-running this queries EFFIS's live database again and OVERWRITES fire/data/catalog.json:
# EFFIS may have added or revised records since this repo's committed snapshot, so diff
# before committing an update, rather than assuming a byte-identical result.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to a finished LEFIRE run's RUN_ROOT, e.g. a run_liaise_ecland.sh output tree}
SURFCLIM=${SURFCLIM:-init_clim/work/surfclim}
OUT_DIR=${OUT_DIR:-fire/work}
START_YEAR=${START_YEAR:-1988}
END_YEAR=${END_YEAR:-2024}
MIN_AREA_HA=${MIN_AREA_HA:-1000}

mkdir -p "$OUT_DIR"

echo "== fetching EFFIS burnt-area records =="
python3 fire/fetch_effis_fires.py --min-area-ha 500 --out "$OUT_DIR/effis_raw.json"

echo "== building the domain fire catalogue =="
python3 fire/build_fire_catalog.py --effis-json "$OUT_DIR/effis_raw.json" --surfclim "$SURFCLIM" \
    --min-area-ha "$MIN_AREA_HA" --end-date "${END_YEAR}-12-31" --out fire/data/catalog.json

echo "== extracting the daily fuel-moisture cache from $RUN_ROOT =="
python3 fire/extract_fire_daily_moisture.py --run-root "$RUN_ROOT" --surfclim "$SURFCLIM" \
    --start-year "$START_YEAR" --end-year "$END_YEAR" --out "$OUT_DIR/fire_daily_moisture.npz"

echo "== drawing the per-event figures =="
python3 fire/plot_fire_event_maps.py --catalog fire/data/catalog.json --cache "$OUT_DIR/fire_daily_moisture.npz" \
    --out-dir "$OUT_DIR/plots" --stats-out "$OUT_DIR/stats.json"

echo "== overview map, timeline and CSV =="
python3 fire/plot_fire_overview.py --catalog fire/data/catalog.json --stats "$OUT_DIR/stats.json" --out-dir "$OUT_DIR"

echo "== trend-controlled baseline check =="
python3 fire/check_fuel_dryness_baseline.py --catalog fire/data/catalog.json --stats "$OUT_DIR/stats.json" \
    --cache "$OUT_DIR/fire_daily_moisture.npz" --out "$OUT_DIR/dryness_baseline_check.json"

echo "done -- see $OUT_DIR"
