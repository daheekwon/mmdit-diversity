#!/bin/bash
# E03 driver (PROTOCOL_ko.md). Usage: nohup bash experiments/E03_klein_port/run.sh <pid> [<pid> ...] &
# 1. wait until every given PID has exited (the image queues started before this one; kill -0, no pgrep)
# 2. V-edit bases (one capture, steps 0-4 and 0-6)
# 3. per split, in order dev, ho3, coco48: generation (Original + 20 grid points, + DAVE if approved by then), then that
#    split's evaluation in the background (value_diversity study_eval + vendi_eval + study_judge -> report + sheets)
#    while the next split generates. "EVAL_DONE <split>" per split, "EVAL_DONE" when all three are evaluated.
# 4. DAVE for the splits generated before experiments/E03_klein_port/DAVE_APPROVED existed, then re-evaluation (ALL_DONE)
# GPUs 0-2 through scripts/lanes.py (memory checks; other sessions share them). GPU 3 is left for the queued SD3.5
# uvr_attention aligned run. Every stage resumes: lanes.py keeps state.json, generation skips finished images.
set -u
cd /NHNHOME/workspace/mmdit-diversity
E=experiments/E03_klein_port
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
L=$E/logs
GPUS="${GPUS:-0 1 2}"   # override: GPUS="1 2 3" bash run.sh
SPLITS="dev ho3 coco48"
mkdir -p $L
log() { echo "$(date '+%F %T') $*" >> $L/driver.log; }
lanes() { $PY scripts/lanes.py "$1" --gpus $GPUS --logs "$2" >> $L/lanes.log 2>&1; }
all_done() { $PY - "$1" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if s and all(v.get('status') == 'done' for v in s.values()) else 1)
EOF
}
evaluate() {   # $1 split, $2 tag (main | dave)
  $PY $E/e03.py link --split $1 >> $L/eval.log 2>&1
  rm -rf $L/eval_$1_$2                                # always re-run: features are cached, the judge resumes
  lanes $E/jobs_eval_$1.txt $L/eval_$1_$2
  all_done $L/eval_$1_$2/state.json || log "evaluation $1 $2 incomplete (see $L/eval_$1_$2)"
  flock $L/report.lock -c "$PY $E/e03.py report > $L/report.log 2>&1; $PY $E/e03.py sheets >> $L/report.log 2>&1"
  log "EVAL_DONE $1 ($2)"
}

log "start; waiting for pids $*"
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 120; done; log "pid $p exited"; done
$PY $E/e03.py jobs >> $L/driver.log 2>&1

if [ ! -f bases/klein_dev8_s0-4_k256.pt ] || [ ! -f bases/klein_dev8_s0-6_k256.pt ]; then
  log "basis capture"
  lanes $E/jobs_basis.txt $L/basis
fi
if [ ! -f bases/klein_dev8_s0-4_k256.pt ] || [ ! -f bases/klein_dev8_s0-6_k256.pt ]; then log "basis missing; stop"; exit 1; fi
log "bases ready"

NODAVE=""
for s in $SPLITS; do
  cp $E/jobs_gen_$s.txt $L/jobs_gen_${s}_run.txt
  if [ -f $E/DAVE_APPROVED ]; then cat $E/jobs_dave_$s.txt >> $L/jobs_gen_${s}_run.txt; else NODAVE="$NODAVE $s"; fi
  log "generation $s (DAVE approved: $([ -f $E/DAVE_APPROVED ] && echo yes || echo no))"
  lanes $L/jobs_gen_${s}_run.txt $L/gen_$s
  all_done $L/gen_$s/state.json || log "generation $s: some jobs not done (see $L/gen_$s/state.json)"
  log "generation $s finished"
  evaluate $s main &
done
wait
log "EVAL_DONE"

if [ -n "$NODAVE" ]; then
  log "DAVE missing for:$NODAVE; waiting for $E/DAVE_APPROVED"
  while [ ! -f $E/DAVE_APPROVED ]; do sleep 600; done
  for s in $NODAVE; do
    lanes $E/jobs_dave_$s.txt $L/dave_$s
    all_done $L/dave_$s/state.json || log "DAVE $s: some jobs not done"
    evaluate $s dave &
  done
  wait
fi
log "ALL_DONE"
