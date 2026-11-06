#!/bin/bash
# E03 COCO48 confirmation (PROTOCOL_ko.md §11; user 11-06 "네, 큐에 넣어"): Original, DAVE alpha 0.2 block 13 steps 0-10,
# ours lambda 1.5 blocks 0-24 steps 0-10 (V + pad V 1/4) on COCO48 only. GPUs 0 and 3 (user 11-05 19:53).
# 1. generation (6 jobs: 3 conditions x 2 prompt shards) -> re-run any run still missing images
# 2. link coco48 -> evaluation (study_eval + vendi + judge) -> report + sheets
# 3. blind sheets for the eye count (blind_count/make_blind_coco48.py); "COCO48_DONE" in driver.log
# Usage: setsid nohup bash experiments/E03_klein_port/run_coco48_confirm.sh &
set -u
cd /NHNHOME/workspace/mmdit-diversity
E=experiments/E03_klein_port
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
L=$E/logs
GPUS="${GPUS:-0 3}"
log() { echo "$(date '+%F %T') $*" >> $L/driver.log; }
lanes() { $PY scripts/lanes.py "$1" --gpus $GPUS --logs "$2" >> $L/lanes.log 2>&1; }
all_done() { $PY - "$1" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if s and all(v.get('status') == 'done' for v in s.values()) else 1)
PYEOF
}

log "COCO48 confirmation: start (Original, DAVE steps 0-10, ours lam 1.5 0-24 steps 0-10; GPUs $GPUS)"
lanes $E/jobs_coco48_confirm.txt $L/gen_coco48_confirm
$PY $E/e03.py missing --split coco48 --jobs $E/jobs_coco48_confirm.txt --out $L/jobs_fix_coco48.txt >> $L/driver.log 2>&1
if [ -s $L/jobs_fix_coco48.txt ]; then
  lanes $L/jobs_fix_coco48.txt $L/gen_fix_coco48
  $PY $E/e03.py missing --split coco48 --jobs $E/jobs_coco48_confirm.txt --out $L/jobs_fix_coco48_2.txt >> $L/driver.log 2>&1
fi
log "COCO48 generation finished"
$PY $E/e03.py link --split coco48 >> $L/eval.log 2>&1
rm -rf $L/eval_coco48
lanes $E/jobs_eval_coco48.txt $L/eval_coco48
all_done $L/eval_coco48/state.json || log "evaluation coco48 incomplete (see $L/eval_coco48)"
flock $L/report.lock -c "$PY $E/e03.py report > $L/report.log 2>&1; $PY $E/e03.py sheets >> $L/report.log 2>&1"
$PY $E/blind_count/make_blind_coco48.py >> $L/report.log 2>&1
log "COCO48_DONE (report + blind sheets; eye count not done yet)"
