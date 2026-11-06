"""Queue part of the 11-05 amendment (PROTOCOL_ko.md §10), appended once to the running queue:
  - SD3.5 E1 extra point v_l1.17 on e06 (images + signal capture) and an evaluation pass 'a' after it,
    chained after pass 2 (the passes write the same files);
  - the combined final setting on ho4 (SD3.5 v_l1.14+padv0.25, FLUX v_l1.72+padv0.25); the ho4 evaluation jobs,
    still pending, now wait for it.
  python experiments/E06_boundary/amend_queue.py
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
COMBO = dict(sd35='v_l1.14+padv0.25', flux='v_l1.72+padv0.25')


def main():
    q = C.OUT/'queue'
    jobs = json.loads((q/'jobs.json').read_text())
    have = {j['id'] for j in jobs}
    new = []
    gen = [f'A_sd35_e06_s{i}' for i in range(2)]
    new += [dict(id=g, mem=GEN_MEM['sd35'], prio=9, cmd=ENV+[PY, RUN, '--model', 'sd35', '--split', 'e06', '--modes', 'v_l1.17',
                                                            '--shard', f'{i}/2', '--gpu', '{gpu}']) for i, g in enumerate(gen)]
    new += [dict(id=f'AS_sd35_s{i}', mem=SIG_MEM['sd35'], prio=13,
                 cmd=ENV+[PY, PROBE, 'signals', '--model', 'sd35', '--modes', 'v_l1.17', '--shard', f'{i}/2', '--gpu', '{gpu}'])
            for i in range(2)]
    new += eval_pass('sd35', 'a', gen, prev='2')
    ho4 = {}
    for m in ('sd35', 'flux'):
        ho4[m] = [f'H_{m}_ho4_s{i}' for i in range(2)]
        new += [dict(id=g, mem=GEN_MEM[m], prio=15, cmd=ENV+[PY, RUN, '--model', m, '--split', 'ho4', '--modes', COMBO[m],
                                                            '--shard', f'{i}/2', '--gpu', '{gpu}']) for i, g in enumerate(ho4[m])]
    new = [j for j in new if j['id'] not in have]
    n = 0
    for j in jobs:
        for m in ('sd35', 'flux'):
            if j['id'] in (f'E_{m}_ho4', f'J_{m}_ho4_0', f'J_{m}_ho4_1'):
                add = [h for h in ho4[m] if h not in j.get('after', [])]
                j['after'] = j.get('after', [])+add
                n += len(add)
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs+new, indent=1))
    tmp.replace(q/'jobs.json')
    print(len(new), 'jobs added:', [j['id'] for j in new], '| ho4 eval deps added:', n)


if __name__ == '__main__':
    main()
