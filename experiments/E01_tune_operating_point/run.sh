#!/usr/bin/env bash
# E01 commands in order (PROTOCOL_ko.md §9). Usage: bash experiments/E01_tune_operating_point/run.sh <stage>
#   coco-sets   write the COCO test and tune caption files (once; COCO 48 images replaced)
#   basis       check the calibration prompts (CALIB=<file>), then fit both bases on GPU lanes
#   tune        generate + evaluate the 11 tune conditions per model and tune set
#   report      section-5 tables (tune_<model>.md) and side-by-side sheets (sheets/)
#   full        generate + evaluate the full-scale runs (only after the section-5 check passed and is recorded)
# GPUs: GPUS="1 2 3" (default). Each GPU runs one job at a time once it has enough free memory (scripts/lanes.py).
set -euo pipefail
E=$(cd "$(dirname "$0")" && pwd)
cd "$E/../.."
PY=${PY:-/DATA0/dahee/diversity/.miniconda3/envs/diversity/bin/python}
GPUS=${GPUS:-"1 2 3"}
lanes() { $PY "$E/e01.py" jobs "$@" > "$E/jobs_$1.txt"; $PY scripts/lanes.py "$E/jobs_$1.txt" --gpus $GPUS --logs "$E/logs/$1"; }
case "${1:-}" in
  coco-sets) $PY "$E/e01.py" coco-sets ;;
  basis)
    : "${CALIB:?set CALIB=<calibration prompt file>}"
    $PY "$E/e01.py" check-calib "$CALIB"
    lanes basis "$CALIB" ;;
  tune) lanes tune ;;
  report) $PY "$E/e01.py" report ;;
  full) lanes full ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
