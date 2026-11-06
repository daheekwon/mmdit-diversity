"""Keeps this experiment's queue off GPUs 0-2 while another session's waiters (value_diversity/results/
uvr_attention_aligned/wait_<model>.log, booked on 2026-11-04 09:53 for GPUs 0-2) have not started. Once every such log
says 'run start' (or 'run exit'), every job of <qdir>/jobs.json that is not done gets gpus 0-3, and this script exits.

  python experiments/E01_res1024_coco48/widen_gpus.py <qdir>
"""
import json
from pathlib import Path
import sys
import time

LOGS = Path(__file__).resolve().parents[3]/'value_diversity'/'results'/'uvr_attention_aligned'


def started():
    logs = [LOGS/f'wait_{m}.log' for m in ('sd35', 'flux', 'klein')]
    return all(p.exists() and ('run start' in p.read_text() or 'run exit' in p.read_text()) for p in logs)


def main(qdir):
    qdir = Path(qdir)
    while not started():
        time.sleep(60)
    state = json.loads((qdir/'state.json').read_text())
    jobs = json.loads((qdir/'jobs.json').read_text())
    for j in jobs:
        if state.get(j['id'], {}).get('status') != 'done':
            j['gpus'] = [0, 1, 2, 3]
    tmp = qdir/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs, indent=1))
    tmp.replace(qdir/'jobs.json')
    print(time.strftime('%H:%M:%S'), 'widened to GPUs 0-3', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
