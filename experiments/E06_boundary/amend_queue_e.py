"""DESIGN_e3_split_ko.md A and B0 (added after the SD3.5 E3 result): text route split + reaction-size signals.

  python experiments/E06_boundary/amend_queue_e.py --null sd35          # null check job only
  python experiments/E06_boundary/amend_queue_e.py --gen sd35           # A: 4 conditions x 50 prompts x 8 seeds, 4 shards
                                                                        #    + eval pass 'e' after the model's last pass
  python experiments/E06_boundary/amend_queue_e.py --b0 sd35            # B0: signals of 4 conditions x 50 prompts x 2 seeds

Generation and B0 jobs wait for the null job. Jobs already in the queue are left as they are.
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from add_stage2 import E3_MEM, PROBE, SIG_MEM, eval_pass                      # noqa: E402
from make_queue import ENV, PY                                                # noqa: E402

NULL_MEM = dict(sd35=14, flux=32)


def null_job(model):
    return dict(id=f'N_split_{model}', mem=NULL_MEM[model], prio=3,
                cmd=ENV+[PY, PROBE, 'nullsplit', '--model', model, '--gpu', '{gpu}'])


def last_pass(jobs, model):
    """Tag of the model's last eval pass in the queue (passes chain on the previous pass's jobs)."""
    tags = [j['id'][1:-len(f'_{model}_e06')] for j in jobs if j['id'].startswith('E') and j['id'].endswith(f'_{model}_e06')]
    return tags[-1]


def gen_jobs(model, jobs):
    lam = C.SPLIT_LAM[model]
    names = [f'{lam}+{s}' for s in C.SPLIT]
    gen = [f'Y_{model}_s{i}' for i in range(4)]
    out = [dict(id=g, mem=E3_MEM[model], prio=12, after=[f'N_split_{model}'],
                cmd=ENV+[PY, PROBE, 'e3', '--model', model, '--modes', *names, '--shard', f'{i}/4', '--gpu', '{gpu}'])
           for i, g in enumerate(gen)]
    return out+eval_pass(model, 'e', gen, prev=last_pass(jobs, model), modes=names)


def b0_jobs(model):
    lam = C.SPLIT_LAM[model]
    names = ['original', lam, f'{lam}+fixtxt', f'{lam}+fixit']
    return [dict(id=f'B0_{model}_s{i}', mem=E3_MEM[model], prio=11, after=[f'N_split_{model}'],
                 cmd=ENV+[PY, PROBE, 'signals', '--model', model, '--modes', *names, '--tag', 'b0', '--text-sig',
                          '--shard', f'{i}/2', '--gpu', '{gpu}']) for i in range(2)]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--null', nargs='*', default=[])
    a.add_argument('--gen', nargs='*', default=[])
    a.add_argument('--b0', nargs='*', default=[])
    x = a.parse_args()
    q = C.OUT/'queue'
    jobs = json.loads((q/'jobs.json').read_text())
    have = {j['id'] for j in jobs}
    new = [null_job(m) for m in dict.fromkeys(x.null+x.gen+x.b0)]
    new += [j for m in x.gen for j in gen_jobs(m, jobs)]+[j for m in x.b0 for j in b0_jobs(m)]
    new = [j for j in new if j['id'] not in have]
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs+new, indent=1))
    tmp.replace(q/'jobs.json')
    print(len(new), 'jobs added:', [j['id'] for j in new])


if __name__ == '__main__':
    main()
