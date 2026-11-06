#!/usr/bin/env bash
# One resumable job of this experiment on the shared GPUs. Waits for a GPU with >= NEED GiB free on two checks 30 s apart
# that no other job of this experiment holds (claim folders in outputs/cogvideox/pilot/claims/, at most MAX jobs at once),
# runs the command with {gpu} replaced, and restarts it in a new process when it fails (out of memory when another
# user's job grows), at most TRIES times. The jobs themselves resume (generate.py skips finished videos;
# fit_basis_resumable.py keeps a checkpoint).
#   bash gpu_job.sh <need GiB> <log name> <command with {gpu}>
set -uo pipefail
NEED=$1; NAME=$2; shift 2; CMD="$*"
MAX=3; TRIES=8
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=$ROOT/outputs/cogvideox/pilot/logs; CLAIMS=$ROOT/outputs/cogvideox/pilot/claims
mkdir -p "$LOG" "$CLAIMS"
free() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "%d %.1f\n", $1, ($3-$2)/1024}'; }
claim() {
  local d=$CLAIMS/gpu$1 p
  for o in "$CLAIMS"/gpu*; do                       # drop claims whose job is gone
    [ -d "$o" ] || continue; p=$(cat "$o/pid" 2>/dev/null)
    { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; } || rm -rf "$o"
  done
  [ "$(ls -d "$CLAIMS"/gpu* 2>/dev/null | wc -l)" -ge $MAX ] && return 1
  mkdir "$d" 2>/dev/null || return 1
  echo $$ > "$d/pid"
}
for try in $(seq 1 $TRIES); do
  gpu=""
  while [ -z "$gpu" ]; do
    a=$(free); sleep 30; b=$(free)
    for g in $(paste <(echo "$a") <(echo "$b") | awk -v n=$NEED '{m=($2<$4)?$2:$4; if (m>=n) print m, $1}' | sort -rn | awk '{print $2}'); do
      if claim "$g"; then gpu=$g; break; fi
    done
    [ -z "$gpu" ] && sleep $((60 + RANDOM % 30))
  done
  echo "$NAME try $try GPU $gpu $(date '+%F %T')"
  ${CMD//\{gpu\}/$gpu} > "$LOG/${NAME}_try$try.log" 2>&1
  rc=$?
  rm -rf "$CLAIMS/gpu$gpu"
  echo "$NAME exit=$rc $(date '+%F %T')"
  [ $rc -eq 0 ] && exit 0
  sleep 60
done
exit 1
