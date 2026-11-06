"""PROTOCOL_ko.md §17 (added after the SD3.5 E3 result): DAVE standard setting with the same two E3 blocks.

  python experiments/E06_boundary/amend_queue_d.py --null sd35        # null check job only
  python experiments/E06_boundary/amend_queue_d.py --gen sd35         # generation (after the null job) + eval pass 'd'

Jobs already in the queue are left as they are. Generation: 2 conditions (dave+fixit, dave+fixtxt) x 50 prompts x 8 seeds,
4 shards; eval pass 'd' chains after the model's last pass (judge only the new conditions).
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from add_stage2 import E3_MEM, PROBE, eval_pass                               # noqa: E402
from make_queue import ENV, PY                                                # noqa: E402
from amend_queue_e4 import last_pass                                           # noqa: E402

NULL_MEM = dict(sd35=14, flux=32)


def null_job(model):
    return dict(id=f'N_dave_{model}', mem=NULL_MEM[model], prio=3,
                cmd=ENV+[PY, PROBE, 'nulldave', '--model', model, '--gpu', '{gpu}'])


def gen_jobs(model, jobs):
    names = [f'{C.E3_DAVE[model]}+fix{x}' for x in ('it', 'txt')]
    gen = [f'D_{model}_s{i}' for i in range(4)]
    out = [dict(id=g, mem=E3_MEM[model], prio=12, after=[f'N_dave_{model}'],
                cmd=ENV+[PY, PROBE, 'e3', '--model', model, '--modes', *names, '--shard', f'{i}/4', '--gpu', '{gpu}'])
           for i, g in enumerate(gen)]
    return out+eval_pass(model, 'd', gen, prev=last_pass(jobs, model), modes=names)   # after the model's last pass


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--null', nargs='*', default=[])
    a.add_argument('--gen', nargs='*', default=[])
    x = a.parse_args()
    q = C.OUT/'queue'
    jobs = json.loads((q/'jobs.json').read_text())
    have = {j['id'] for j in jobs}
    new = [null_job(m) for m in x.null+x.gen]+[j for m in x.gen for j in gen_jobs(m, jobs)]
    new = [j for j in new if j['id'] not in have]
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs+new, indent=1))
    tmp.replace(q/'jobs.json')
    print(len(new), 'jobs added:', [j['id'] for j in new])


if __name__ == '__main__':
    main()
