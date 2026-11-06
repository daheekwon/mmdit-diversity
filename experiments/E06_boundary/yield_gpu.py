"""Keep the E06 queue off one GPU until E04's last FLUX 1024 generation jobs are done (request from another session,
2026-11-05 20:1x: G1024_flux_s6 / s7 need about 37 GB free on one GPU). Running E06 jobs are not touched; only jobs
that have not started get `gpus` without the reserved GPU. When both E04 jobs (or their requeued <id>b copies) are done,
the restriction is removed and the script exits.

  python experiments/E06_boundary/yield_gpu.py <e06 qdir> <e04 qdir> <gpu>
"""
import json
from pathlib import Path
import sys
import time

E06, E04, GPU = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
WAIT = ('G1024_flux_s6', 'G1024_flux_s7')
ALL = [0, 1, 2, 3]
KEEP = [g for g in ALL if g != GPU]


def e04_done():
    st = json.loads((E04/'state.json').read_text())
    return all(any(st.get(i, {}).get('status') == 'done' for i in (j, j+'b')) for j in WAIT)


def set_gpus(restrict):
    st = json.loads((E06/'state.json').read_text()) if (E06/'state.json').exists() else {}
    jobs = json.loads((E06/'jobs.json').read_text())
    changed = 0
    for j in jobs:
        if st.get(j['id'], {}).get('status') in ('running', 'done'):
            continue
        want = KEEP if restrict else ALL
        if j.get('gpus') != want:
            j['gpus'] = want
            changed += 1
    if changed:
        tmp = E06/'jobs.json.yield'
        tmp.write_text(json.dumps(jobs, indent=1))
        tmp.replace(E06/'jobs.json')
    return changed


print(time.strftime('%H:%M:%S'), 'reserve gpu', GPU, 'for', WAIT, flush=True)
while not e04_done():
    n = set_gpus(True)
    if n:
        print(time.strftime('%H:%M:%S'), 'restricted', n, 'jobs to', KEEP, flush=True)
    time.sleep(60)
print(time.strftime('%H:%M:%S'), 'E04 jobs done; released gpu', GPU, 'jobs', set_gpus(False), flush=True)
