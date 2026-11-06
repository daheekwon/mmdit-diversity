"""Run shell jobs on shared GPUs, one job per GPU at a time.

  python scripts/lanes.py jobs.txt --gpus 1 2 3 --logs logs/tune

jobs.txt: one job per line, "<GiB free needed><TAB><shell command with {gpu}>"; empty and '#' lines are skipped.
Each GPU takes the first waiting job whose memory need it meets on two checks 30 s apart (other sessions use the same
GPUs). A job that fails with a CUDA out-of-memory error goes back to the queue (up to 5 tries; generation skips
finished images). Other failures are recorded and not retried. <logs>/state.json keeps each job's status, so a
restart skips the jobs already done.
"""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

OOM = ('OutOfMemoryError', 'CUDA out of memory')
TRIES = 5


def free_gib(gpu):
    try:
        out = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60).stdout
        used, total = map(float, out.split(','))
        return (total-used)/1024
    except (ValueError, subprocess.TimeoutExpired):
        return 0.


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('jobs')
    a.add_argument('--gpus', type=int, nargs='+', required=True)
    a.add_argument('--logs', required=True)
    x = a.parse_args()
    logs = Path(x.logs)
    logs.mkdir(parents=True, exist_ok=True)
    state_path = logs/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    jobs = []
    for line in Path(x.jobs).read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            need, cmd = line.split('\t', 1)
            jobs.append(dict(need=float(need), cmd=cmd))
    waiting = [j for j in jobs if state.get(j['cmd'], {}).get('status') != 'done']
    lock = threading.Lock()
    print(f'{len(jobs)} jobs, {len(jobs)-len(waiting)} already done, GPUs {x.gpus}', flush=True)

    def save():
        state_path.write_text(json.dumps(state, indent=1))

    def lane(gpu):
        while True:
            with lock:
                if not waiting:
                    return
                free = free_gib(gpu)
                job = next((j for j in waiting if j['need'] <= free), None)
                if job:
                    waiting.remove(job)
            if job is None:
                time.sleep(60)
                continue
            time.sleep(30)                                   # second memory check
            if free_gib(gpu) < job['need']:
                with lock:
                    waiting.insert(0, job)
                time.sleep(60)
                continue
            with lock:
                s = state.setdefault(job['cmd'], dict(tries=0))
                s.update(status='running', gpu=gpu, started=time.time(), tries=s['tries']+1)
                log = logs/f'{list(state).index(job["cmd"]):03d}.{s["tries"]}.log'
                s['log'] = str(log)
                save()
            print(f'[GPU {gpu}] start {log.name}: {job["cmd"][:150]}', flush=True)
            with open(log, 'w') as fh:
                rc = subprocess.run(job['cmd'].format(gpu=gpu), shell=True, stdout=fh, stderr=subprocess.STDOUT).returncode
            oom = rc != 0 and any(o in log.read_text(errors='replace') for o in OOM)
            with lock:
                s.update(rc=rc, ended=time.time(), status='done' if rc == 0 else 'oom' if oom else 'failed')
                if oom and s['tries'] < TRIES:
                    waiting.append(job)
                save()
            print(f'[GPU {gpu}] {s["status"]} (rc {rc}) {log.name}', flush=True)

    threads = [threading.Thread(target=lane, args=(g,)) for g in x.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [c for c, s in state.items() if s.get('status') != 'done']
    print(f'finished: {len(state)-len(bad)} done, {len(bad)} not done', flush=True)
    for c in bad:
        print(f'  {state[c]["status"]}: {state[c]["log"]}', flush=True)


if __name__ == '__main__':
    main()
