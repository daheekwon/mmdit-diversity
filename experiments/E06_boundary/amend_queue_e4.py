"""PROTOCOL_ko.md §20: E4 guard (FLUX only; the E3 rule fired for pad 1/4 with both blocks).

  python experiments/E06_boundary/amend_queue_e4.py flux

Jobs (already-queued ids are left as they are):
  N_e4_<m>      null check (null_e4.py)
  U_<m>_s0-1    unguarded combination upper point (v_l2.01+padv0.25), run_e06.py with the package Controller
  W_<m>_s0-7    guarded combination: 3 lambdas x 2 blocks, probe e3, after the null job
  eval pass '4' after the model's last eval pass in the queue (judge only the new conditions)
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from add_stage2 import E3_MEM, PROBE, eval_pass                               # noqa: E402
from make_queue import ENV, GEN_MEM, PY                                       # noqa: E402

NULL_MEM = dict(sd35=14, flux=32)
RUN, NULL = str(HERE/'run_e06.py'), str(HERE/'null_e4.py')


def last_pass(jobs, model):
    """Tag of the model's last eval pass in the queue (passes write the same files, so each waits for the previous)."""
    tags = [j['id'][1:-len(f'_{model}_e06')] for j in jobs if j['id'].startswith('E') and j['id'].endswith(f'_{model}_e06')]
    return tags[-1]


def jobs_for(model, jobs):
    plain, guarded = C.e4_names(model)
    if not guarded:
        raise SystemExit(f'{model}: no E4 conditions (common.E4_LAM)')
    out = [dict(id=f'N_e4_{model}', mem=NULL_MEM[model], prio=3,
                cmd=ENV+[PY, NULL, '--model', model, '--gpu', '{gpu}'])]
    u = [f'U_{model}_s{i}' for i in range(2)]
    out += [dict(id=g, mem=GEN_MEM[model], prio=9,
                 cmd=ENV+[PY, RUN, '--model', model, '--split', 'e06', '--modes', *plain, '--shard', f'{i}/2',
                          '--gpu', '{gpu}']) for i, g in enumerate(u)]
    w = [f'W_{model}_s{i}' for i in range(8)]
    out += [dict(id=g, mem=E3_MEM[model], prio=12, after=[f'N_e4_{model}'],
                 cmd=ENV+[PY, PROBE, 'e3', '--model', model, '--modes', *guarded, '--shard', f'{i}/8', '--gpu', '{gpu}'])
            for i, g in enumerate(w)]
    return out+eval_pass(model, '4', u+w, prev=last_pass(jobs, model), modes=plain+guarded)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('models', nargs='+')
    x = a.parse_args()
    q = C.OUT/'queue'
    jobs = json.loads((q/'jobs.json').read_text())
    have = {j['id'] for j in jobs}
    new = [j for m in x.models for j in jobs_for(m, jobs) if j['id'] not in have]
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs+new, indent=1))
    tmp.replace(q/'jobs.json')
    print(len(new), 'jobs added:', [(j['id'], j.get('after')) for j in new])


if __name__ == '__main__':
    main()
