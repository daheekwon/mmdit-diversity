"""Pause / resume the FLUX 1024 generation jobs of this experiment's queue (outputs/res1024/queue/jobs.json).

Both job queues on this machine launch by a free-memory snapshot, so when they start on the same GPU in the same
minute one of them runs out of memory and spends a retry (at most 6, then the job is failed and D1024_flux never runs).
  pause:  gpus = [] for every unfinished G_flux_* job (the queue skips them, tries are kept).
  resume: wait until every F2_* job of the sx_coco queue is done or failed, then gpus = [0, 1, 2, 3].
The other queue's path is fixed here, not in argv (see the waiter pgrep rule).
"""
import json
from pathlib import Path
import sys
import time

QDIR = Path(__file__).resolve().parents[2]/'outputs'/'res1024'/'queue'
OTHER = Path('/DATA0/dahee/diversity_migration/value_diversity/results/generalize/sx_coco/queue')


def set_gpus(gpus):
    state = json.loads((QDIR/'state.json').read_text())
    jobs = json.loads((QDIR/'jobs.json').read_text())
    n = 0
    for j in jobs:
        if j['id'].startswith('G_flux_') and state.get(j['id'], {}).get('status') != 'done':
            j['gpus'] = gpus
            n += 1
    tmp = QDIR/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs, indent=1))
    tmp.replace(QDIR/'jobs.json')
    print(time.strftime('%H:%M:%S'), 'G_flux jobs', n, 'gpus', gpus, flush=True)


def other_gen_left():
    jobs = json.loads((OTHER/'jobs.json').read_text())
    state = json.loads((OTHER/'state.json').read_text())
    return [j['id'] for j in jobs if j['id'].startswith('F2_')
            and state.get(j['id'], {}).get('status') not in ('done', 'failed')]


if __name__ == '__main__':
    if sys.argv[1] == 'pause':
        set_gpus([])
    else:
        while left := other_gen_left():
            print(time.strftime('%H:%M:%S'), 'other queue generation left', len(left), flush=True)
            time.sleep(120)
        set_gpus([0, 1, 2, 3])
