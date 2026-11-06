#!/usr/bin/env bash
# Section 7e (user request 2026-11-05 "움직임 측정"): RAFT motion of the six 16 x 8 runs (div/video_metrics.py 'flow'),
# at most 3 jobs at once through gpu_job.sh; then the paired tables in outputs/cogvideox/pilot/motion_compare.md.
#   nohup bash experiments/E02_t2v_cogvideox_pilot/run_motion.sh > outputs/cogvideox/pilot/logs/motion_driver.log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
OUT=$ROOT/outputs/cogvideox/vbench_tune
cd "$ROOT"
echo "motion start $(date '+%F %T')"
for t in original ours_time_mean v_edit_time_mean ours v_edit pad_v; do
  bash "$HERE/gpu_job.sh" 6 "flow_$t" $PY scripts/evaluate_video.py "$OUT/$t" --metrics flow --device cuda:{gpu} &
  sleep 40
done
wait
M="--metrics motion dynamic flicker_mae subject_consistency"
{
  echo "## vs original"; $PY scripts/compare.py "$OUT/original" "$OUT/ours_time_mean" "$OUT/v_edit_time_mean" "$OUT/ours" "$OUT/v_edit" "$OUT/pad_v" $M
  echo; echo "## time mean vs plain (ours)"; $PY scripts/compare.py "$OUT/ours" "$OUT/ours_time_mean" $M
  echo; echo "## time mean vs plain (V edit only)"; $PY scripts/compare.py "$OUT/v_edit" "$OUT/v_edit_time_mean" $M
} > "$ROOT/outputs/cogvideox/pilot/motion_compare.md" 2>&1
echo "ALL_DONE $(date '+%F %T')"
