"""Watches a combo_queue folder: a job whose retries ran out is appended again as <id>b (same command), and every job
that waited for it now waits for the copy. Stops when the queue log says QUEUE_EMPTY.
  python experiments/E04_res1024_transfer/requeue_failed.py <qdir>"""
import json
from pathlib import Path
import sys
import time

q = Path(sys.argv[1])
while True:
    time.sleep(120)
    if 'QUEUE_EMPTY' in (q/'nohup.out').read_text(errors='ignore')[-200:]:
        break
    state = json.loads((q/'state.json').read_text()) if (q/'state.json').exists() else {}
    jobs = json.loads((q/'jobs.json').read_text())
    ids = {j['id'] for j in jobs}
    failed = [k for k, s in state.items() if s.get('status') == 'failed' and k+'b' not in ids]
    if not failed:
        continue
    for k in failed:
        j = dict(next(j for j in jobs if j['id'] == k), id=k+'b')
        jobs.append(j)
        for o in jobs:
            if k in o.get('after', []):
                o['after'] = [k+'b' if d == k else d for d in o['after']]
        print(time.strftime('%H:%M:%S'), 'REQUEUED', k, '->', k+'b', flush=True)
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs, indent=1))
    tmp.replace(q/'jobs.json')
