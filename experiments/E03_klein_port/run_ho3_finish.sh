#!/bin/bash
# E03 from the middle of ho3 generation on (11-05 17:50, user via the planning session: "ho까지만 볼게" = no COCO48 for
# now; use GPU 0 too). Replaces run.sh, which was stopped together with its ho3 lanes; the ho3 jobs that were running
# then keep running on their own (their generate.py PIDs are the arguments, waited for with kill -0).
# 1. the ho3 jobs not yet started -> lanes on GPUs 0-3 (memory checks as before)
# 2. wait for the given PIDs; any ho3 run still missing images (failed or OOM without retry) -> lanes once more
# 3. evaluate ho3 (same steps as run.sh evaluate) -> report + sheets; "EVAL_DONE ho3 (main)", "ALL_DONE" in driver.log
# Usage: setsid nohup bash experiments/E03_klein_port/run_ho3_finish.sh <pid> [<pid> ...] &
set -u
cd /DATA0/dahee/diversity_migration/mmdit_diversity
E=experiments/E03_klein_port
PY=/DATA0/dahee/diversity/.miniconda3/envs/diversity/bin/python
L=$E/logs
GPUS="${GPUS:-0 1 2 3}"
log() { echo "$(date '+%F %T') $*" >> $L/driver.log; }
lanes() { $PY scripts/lanes.py "$1" --gpus $GPUS --logs "$2" >> $L/lanes.log 2>&1; }
all_done() { $PY - "$1" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if s and all(v.get('status') == 'done' for v in s.values()) else 1)
EOF
}

log "ho3 finish: start (run.sh stopped, COCO48 skipped by user; GPUs $GPUS; waiting also for pids $*)"
$PY - $L/gen_ho3/state.json $L/jobs_gen_ho3_run.txt $L/jobs_gen_ho3_rest.txt <<'EOF'
import json, sys
state = json.load(open(sys.argv[1]))
rest = [l for l in open(sys.argv[2]) if l.strip() and l.split('\t', 1)[1].rstrip('\n') not in state]
open(sys.argv[3], 'w').writelines(rest)
print(len(rest), 'ho3 jobs not started by run.sh')
EOF
lanes $L/jobs_gen_ho3_rest.txt $L/gen_ho3_rest
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
$PY $E/e03.py missing --split ho3 --jobs $L/jobs_gen_ho3_run.txt --out $L/jobs_gen_ho3_fix.txt >> $L/driver.log 2>&1
if [ -s $L/jobs_gen_ho3_fix.txt ]; then
  lanes $L/jobs_gen_ho3_fix.txt $L/gen_ho3_fix
  $PY $E/e03.py missing --split ho3 --jobs $L/jobs_gen_ho3_run.txt --out $L/jobs_gen_ho3_fix2.txt >> $L/driver.log 2>&1
fi
log "generation ho3 finished"

$PY $E/e03.py link --split ho3 >> $L/eval.log 2>&1
rm -rf $L/eval_ho3_main
lanes $E/jobs_eval_ho3.txt $L/eval_ho3_main
all_done $L/eval_ho3_main/state.json || log "evaluation ho3 main incomplete (see $L/eval_ho3_main)"
flock $L/report.lock -c "$PY $E/e03.py report > $L/report.log 2>&1; $PY $E/e03.py sheets >> $L/report.log 2>&1"
log "EVAL_DONE ho3 (main)"
log "ALL_DONE (dev + ho3; COCO48 not run, user 11-05)"
