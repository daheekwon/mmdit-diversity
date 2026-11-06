"""Queue part of the §13 amendment (PROTOCOL_ko.md), appended once to the running queue: the combination path point
below the final lambda (SD3.5 v_l1.07+padv0.25, FLUX v_l1.43+padv0.25) on e06, its signal capture (SD3.5; the pending FLUX
signal jobs pick it up themselves), and evaluation pass 'c' (after pass 'b').
The FLUX generation waits for every FLUX stage-1 shard: the shards that start after this edit already include the new
points, so this job only fills the prompts of the shard that was running.
  python experiments/E06_boundary/amend_queue_c.py
"""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from make_queue import ENV, GEN_MEM, PY                                       # noqa: E402
from add_stage2 import PROBE, SIG_MEM, eval_pass                              # noqa: E402

RUN = str(HERE/'run_e06.py')


def main():
    q = C.OUT/'queue'
    jobs = json.loads((q/'jobs.json').read_text())
    have = {j['id'] for j in jobs}
    new = []
    for m, prev in (('sd35', 'b'), ('flux', 'b')):
        after = [] if m == 'sd35' else [j['id'] for j in jobs if j['id'].startswith('G_flux_e06_s')]
        gen = [f'C_{m}_e06_s{i}' for i in range(2)]
        names = [f'v_l{C.COMBO_LOW[m]}+padv0.25']
        new += [dict(id=g, mem=GEN_MEM[m], prio=9, after=after,
                     cmd=ENV+[PY, RUN, '--model', m, '--split', 'e06', '--modes', *names, '--shard', f'{i}/2',
                              '--gpu', '{gpu}']) for i, g in enumerate(gen)]
        if m == 'sd35':
            new += [dict(id=f'CS_sd35_s{i}', mem=SIG_MEM['sd35'], prio=13,
                         cmd=ENV+[PY, PROBE, 'signals', '--model', 'sd35', '--modes', *names, '--shard', f'{i}/2',
                                  '--gpu', '{gpu}']) for i in range(2)]
        new += eval_pass(m, 'c', gen, prev=prev)
    new = [j for j in new if j['id'] not in have]
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs+new, indent=1))
    tmp.replace(q/'jobs.json')
    print(len(new), 'jobs added:', [j['id'] for j in new])


if __name__ == '__main__':
    main()
