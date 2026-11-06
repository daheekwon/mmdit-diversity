#!/bin/bash
# E03 ho3, rest of generation on two GPUs only (11-05 19:53, user via the planning session: "응, 2개로 줄여" = give
# GPUs 1 and 2 to E04 FLUX 1024 / E06). Replaces run_ho3_finish.sh, which was stopped together with its lanes at 19:53;
# the klein jobs it had started keep running on their own (their generate.py PIDs are the arguments, waited for with
# kill -0; two of them run on GPUs 1 and 2 until they end, no new klein job goes there).
# 1. ho3 jobs started by neither run.sh nor run_ho3_finish.sh -> lanes on GPUs 0 and 3 (memory checks as before)
# 2. wait for the given PIDs; any ho3 run still missing images -> lanes once more (GPUs 0 and 3)
# 3. evaluate ho3 on GPUs 0 and 3 -> report + sheets; "EVAL_DONE ho3 (main)", "ALL_DONE" in driver.log
# Usage: setsid nohup bash experiments/E03_klein_port/run_ho3_finish2.sh <pid> [<pid> ...] &
set -u
cd /NHNHOME/workspace/mmdit-diversity
E=experiments/E03_klein_port
PY=/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python
L=$E/logs
GPUS="${GPUS:-0 3}"
log() { echo "$(date '+%F %T') $*" >> $L/driver.log; }
lanes() { $PY scripts/lanes.py "$1" --gpus $GPUS --logs "$2" >> $L/lanes.log 2>&1; }
all_done() { $PY - "$1" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if s and all(v.get('status') == 'done' for v in s.values()) else 1)
EOF
}

log "ho3 finish 2: start (run_ho3_finish.sh stopped, user: two GPUs only; GPUs $GPUS; waiting also for pids $*)"
$PY - $L/jobs_gen_ho3_run.txt $L/jobs_gen_ho3_rest2.txt $L/gen_ho3/state.json $L/gen_ho3_rest/state.json <<'EOF'
import json, sys
started = set()
for p in sys.argv[3:]:
    started |= set(json.load(open(p)))
rest = [l for l in open(sys.argv[1]) if l.strip() and l.split('\t', 1)[1].rstrip('\n') not in started]
open(sys.argv[2], 'w').writelines(rest)
print(len(rest), 'ho3 jobs not started yet')
EOF
lanes $L/jobs_gen_ho3_rest2.txt $L/gen_ho3_rest2
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
