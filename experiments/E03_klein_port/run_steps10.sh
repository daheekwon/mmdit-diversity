#!/bin/bash
# E03 steps 0-10 first (11-05 20:00, user): ours = V edit lam 1.3 / 1.5 / 1.7 on all blocks 0-24 + pad V 1/4 on all
# blocks, both on steps 0-10; DAVE alpha 0.2 block 13 on steps 0-10. dev + ho3, before the rest of the ho3 grid.
# DAVE steps 0-19 is not run on ho3 any more (user 20:05, "큐에 있다면 없애줘"; dev already has it).
# GPUs 0 and 3 only (user 19:53). Replaces run_ho3_finish2.sh (stopped 20:05 before its DAVE job started); the klein
# jobs it and earlier runners had started keep running on their own (their PIDs are the arguments, waited for before
# the missing-image check).
# 1. basis for steps 0-10 + DAVE steps 0-10 (ho3, dev; DAVE needs no basis)
# 2. ours steps 0-10 (dev, ho3) -> evaluate dev + ho3 so far -> report + sheets; "EVAL_DONE steps 0-10" in driver.log
# 3. rest of the ho3 grid (steps 0-4 / 0-6, no DAVE) -> wait for the PIDs -> re-run any run still missing images
# 4. evaluate dev + ho3 -> report + sheets; "EVAL_DONE ho3 (main)", "ALL_DONE" in driver.log
# Usage: setsid nohup bash experiments/E03_klein_port/run_steps10.sh <pid> [<pid> ...] &
set -u
cd /DATA0/dahee/diversity_migration/mmdit_diversity
E=experiments/E03_klein_port
PY=/DATA0/dahee/diversity/.miniconda3/envs/diversity/bin/python
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
evaluate() {   # $1 = lanes state folder
  $PY $E/e03.py link --split dev >> $L/eval.log 2>&1
  $PY $E/e03.py link --split ho3 >> $L/eval.log 2>&1
  cat $E/jobs_eval_dev.txt $E/jobs_eval_ho3.txt > $L/jobs_eval_dev_ho3.txt
  rm -rf $L/$1
  lanes $L/jobs_eval_dev_ho3.txt $L/$1
  all_done $L/$1/state.json || log "evaluation $1 incomplete (see $L/$1)"
  flock $L/report.lock -c "$PY $E/e03.py report > $L/report.log 2>&1; $PY $E/e03.py sheets >> $L/report.log 2>&1"
}

log "steps 0-10: start (user 20:00; DAVE steps 0-19 dropped from ho3; GPUs $GPUS; waiting also for pids $*)"
cat $E/jobs_basis_long.txt $E/jobs_dave10_ho3.txt $E/jobs_dave10_dev.txt > $L/jobs_s10_phase1.txt
lanes $L/jobs_s10_phase1.txt $L/s10_phase1
if [ -f bases/klein_dev8_s0-10_k256.pt ]; then
  log "steps 0-10: basis ready, DAVE steps 0-10 done; ours steps 0-10 next"
  cat $E/jobs_long_dev.txt $E/jobs_long_ho3.txt > $L/jobs_s10_ours.txt
  lanes $L/jobs_s10_ours.txt $L/s10_ours
else
  log "steps 0-10: basis missing (see $L/s10_phase1); ours steps 0-10 skipped"
fi
evaluate eval_s10
log "EVAL_DONE steps 0-10 (dev + ho3 so far)"

# rest of the ho3 grid: the jobs run_ho3_finish2.sh had not started, without DAVE steps 0-19
$PY - $L/jobs_gen_ho3_rest2.txt $L/jobs_gen_ho3_grid_rest.txt $L/gen_ho3_rest2/state.json <<'EOF'
import json, sys
started = set(json.load(open(sys.argv[3])))
rest = [l for l in open(sys.argv[1]) if l.strip() and l.split('\t', 1)[1].rstrip('\n') not in started
        and '--name dave_a0.2 ' not in l]
open(sys.argv[2], 'w').writelines(rest)
print(len(rest), 'ho3 grid jobs left')
EOF
log "ho3 grid rest: $(grep -c . $L/jobs_gen_ho3_grid_rest.txt) jobs"
lanes $L/jobs_gen_ho3_grid_rest.txt $L/gen_ho3_grid_rest
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done
cat $E/jobs_gen_ho3.txt $E/jobs_long_ho3.txt $E/jobs_dave10_ho3.txt > $L/jobs_check_ho3.txt
cat $E/jobs_long_dev.txt $E/jobs_dave10_dev.txt > $L/jobs_check_dev.txt
fix() {   # $1 = suffix
  $PY $E/e03.py missing --split ho3 --jobs $L/jobs_check_ho3.txt --out $L/jobs_fix_ho3$1.txt >> $L/driver.log 2>&1
  $PY $E/e03.py missing --split dev --jobs $L/jobs_check_dev.txt --out $L/jobs_fix_dev$1.txt >> $L/driver.log 2>&1
  cat $L/jobs_fix_ho3$1.txt $L/jobs_fix_dev$1.txt > $L/jobs_fix$1.txt
}
fix ""
if [ -s $L/jobs_fix.txt ]; then
  lanes $L/jobs_fix.txt $L/gen_fix
  fix 2
fi
log "generation ho3 finished"
evaluate eval_final
log "EVAL_DONE ho3 (main)"
log "ALL_DONE (dev + ho3 with steps 0-10; COCO48 not run, user 11-05)"
