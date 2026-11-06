#!/usr/bin/env bash
# After run_stage2a.sh: video metrics (DINO, MUSIQ, judge) of the four runs, then the paired comparison with Original.
# Waits for ALL_DONE in the stage-2a driver log (a file check; no process search).
#   bash experiments/E02_t2v_cogvideox_pilot/run_stage2a_eval.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
OUT=$ROOT/outputs/cogvideox/vbench_tune
PILOT=$ROOT/outputs/cogvideox/pilot
until grep -q ALL_DONE "$PILOT/logs/stage2a_driver.log" 2>/dev/null; do sleep 300; done
echo "eval start $(date '+%F %T')"
cd "$ROOT"
for t in original pad_v ours v_edit; do
  bash "$HERE/gpu_job.sh" 22 "eval_$t" $PY scripts/evaluate_video.py "$OUT/$t" --device cuda:{gpu} &
  sleep 100
done
wait
$PY scripts/compare.py "$OUT/original" "$OUT/ours" "$OUT/v_edit" "$OUT/pad_v" > "$PILOT/stage2a_compare.md" 2>&1
echo "EVAL_DONE $(date '+%F %T')"
