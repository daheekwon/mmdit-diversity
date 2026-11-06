#!/usr/bin/env bash
# Stage 2, first part (user decisions 2026-11-04, PROTOCOL_ko.md section 7): V-edit basis on the first 32 calibration
# prompts x 8 seeds; 16 tune prompts x 8 seeds for Original, pad V only, ours, V edit only (long prompts).
# Original and pad V only start at once; ours and V edit only start when the basis exists.
#   bash experiments/E02_t2v_cogvideox_pilot/run_stage2a.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
J="bash $HERE/gpu_job.sh"
GEN="$PY $ROOT/scripts/generate.py --model cogvideox --dataset vbench_tune --n-prompts 16 --samples 8 --gpu {gpu}"
cd "$ROOT"
echo "stage2a start $(date '+%F %T')"
$J 20 basis $PY $HERE/fit_basis_resumable.py --name cogvideox_pilot_k256 \
  --prompts /NHNHOME/workspace/data/vbench/calib_dimension_longer_32.txt --seeds 8 --gpu {gpu} &
pb=$!
sleep 100
$J 13 gen_original $GEN --preset original &
sleep 100
$J 13 gen_pad_v $GEN --preset pad_v &
wait $pb; rb=$?
echo "basis exit=$rb $(date '+%F %T')"
if [ $rb -eq 0 ]; then
  $J 13 gen_ours $GEN --preset ours &
  sleep 100
  $J 13 gen_v_edit $GEN --preset v_edit &
fi
wait
echo "ALL_DONE $(date '+%F %T')"
