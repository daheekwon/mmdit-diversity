#!/bin/bash
# E03 dev additions (user request 11-05, after dev generation of the first grid had started; PROTOCOL_ko.md §4):
#   blocks 0-24 (all blocks) x lam 1.3-1.7 x steps 0-4 / 0-6 with pad V 1/4 (10 conditions; ho3 / COCO48 get them
#   through the driver's job lists), and each part of ours alone on dev: pad V 1/4 only (steps 0-4, 0-6) and
#   V edit lam 1.7 blocks 5-24 only (steps 0-4, 0-6).
# 1. generation through lanes.py (memory checks; runs beside the driver, both resume from state.json)
# 2. waits for the driver's "EVAL_DONE dev" line (dev evaluation writes the same csv files), then evaluates dev
#    again (features cached, judge resumes) and rewrites the report + sheets. Logs "EXTRA_DEV_DONE" in driver.log.
# Usage: GPUS="0 1 2 3" setsid nohup bash experiments/E03_klein_port/run_extra_dev.sh &
set -u
cd /NHNHOME/workspace/mmdit-diversity
E=experiments/E03_klein_port
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
L=$E/logs
GPUS="${GPUS:-0 1 2 3}"
log() { echo "$(date '+%F %T') extra dev: $*" >> $L/driver.log; }
lanes() { $PY scripts/lanes.py "$1" --gpus $GPUS --logs "$2" >> $L/lanes_extra.log 2>&1; }

log "start (GPUs $GPUS)"
lanes $E/jobs_extra_dev.txt $L/gen_extra_dev
log "generation finished"
until grep -q "EVAL_DONE dev" $L/driver.log; do sleep 300; done
$PY $E/e03.py link --split dev >> $L/eval.log 2>&1
rm -rf $L/eval_dev_extra
lanes $E/jobs_eval_dev.txt $L/eval_dev_extra
flock $L/report.lock -c "$PY $E/e03.py report > $L/report.log 2>&1; $PY $E/e03.py sheets >> $L/report.log 2>&1"
log "EXTRA_DEV_DONE"
