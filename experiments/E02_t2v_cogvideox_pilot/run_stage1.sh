#!/usr/bin/env bash
# Stage 1 of PROTOCOL_ko.md, one GPU, in order: parity -> cost -> capture -> padmass. A failing step stops the rest.
#   bash experiments/E02_t2v_cogvideox_pilot/run_stage1.sh <gpu>
set -euo pipefail
GPU=${1:?gpu}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
LOG=$ROOT/outputs/cogvideox/pilot/logs
mkdir -p "$LOG"
cd "$ROOT"
echo "stage1 start $(date '+%F %T') gpu $GPU"
$PY tests/parity_cogvideox.py --gpu "$GPU" > "$LOG/parity.log" 2>&1
echo "parity done $(date '+%T')"
for c in cost capture padmass; do
  $PY experiments/E02_t2v_cogvideox_pilot/measure.py $c --gpu "$GPU" > "$LOG/$c.log" 2>&1
  echo "$c done $(date '+%T')"
done
echo "ALL_DONE $(date '+%F %T')"
