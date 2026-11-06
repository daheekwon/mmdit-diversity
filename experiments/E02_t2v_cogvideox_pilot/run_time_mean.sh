#!/usr/bin/env bash
# Section 7c (user request 2026-11-05 "방법 1 시험"): ours and V edit only with the time-mean V edit, 16 tune prompts x
# 8 seeds (same as stage 2a), 3 shards each through gpu_job.sh (at most 3 jobs at once); then the video metrics, the
# comparison tables, the sheets and the blind panels.
#   nohup bash experiments/E02_t2v_cogvideox_pilot/run_time_mean.sh > outputs/cogvideox/pilot/logs/time_mean_driver.log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=/DATA0/dahee/diversity/.miniconda3/envs/diversity/bin/python
J="bash $HERE/gpu_job.sh"
GEN="$PY $ROOT/scripts/generate.py --model cogvideox --dataset vbench_tune --n-prompts 16 --samples 8 --gpu {gpu}"
OUT=$ROOT/outputs/cogvideox/vbench_tune
PILOT=$ROOT/outputs/cogvideox/pilot
cd "$ROOT"
echo "time_mean start $(date '+%F %T')"
for p in ours_time_mean v_edit_time_mean; do
  for s in 0 1 2; do
    $J 13 "gen_${p}_$s" $GEN --preset $p --shard $s/3 &
    sleep 100
  done
done
wait
echo "GEN_DONE $(date '+%F %T')"
for t in ours_time_mean v_edit_time_mean; do
  $J 22 "eval_$t" $PY scripts/evaluate_video.py "$OUT/$t" --device cuda:{gpu} &
  sleep 100
done
wait
{
  echo "## vs original"; $PY scripts/compare.py "$OUT/original" "$OUT/ours_time_mean" "$OUT/v_edit_time_mean" "$OUT/ours" "$OUT/v_edit"
  echo; echo "## time mean vs plain (ours)"; $PY scripts/compare.py "$OUT/ours" "$OUT/ours_time_mean"
  echo; echo "## time mean vs plain (V edit only)"; $PY scripts/compare.py "$OUT/v_edit" "$OUT/v_edit_time_mean"
} > "$PILOT/time_mean_compare.md" 2>&1
$PY scripts/video_sheet.py "$OUT/original" "$OUT/ours" "$OUT/ours_time_mean" "$OUT/v_edit" "$OUT/v_edit_time_mean" \
  --names 원본 "우리 방법" "우리 방법(시간 평균)" V편집만 "V편집만(시간 평균)" --out "$PILOT/sheets_time_mean"
$PY "$HERE/blind_panels.py" make
echo "ALL_DONE $(date '+%F %T')"
