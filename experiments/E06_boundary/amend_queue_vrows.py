"""DESIGN_vrows_ko.md (PROTOCOL §24): image-rows V edit (v_edit.rows=image) vs the current all-rows edit.

  python experiments/E06_boundary/amend_queue_vrows.py sd35 flux

Jobs per model (already-queued ids are left as they are):
  N_vrows_<m>   null checks (null_vrows.py)
  VROW_<m>_s0-3 generation of common.vrows_new(model) with run_e06.py, after the null job; FLUX also waits for the E4
                generation (W_flux_s*, U_flux_s*) so that E4 is not slowed
  eval pass 'r' after the model's last eval pass in the queue (judge only the new conditions)
A model whose pass 'r' is already queued is skipped. SD3.5 was queued 11-06 11:18 as R_sd35_s2-3 + VROW_sd35_s0-1 (the ids
R_<m>_s0-1 already belonged to the random-direction jobs, so the first two shards were renamed by hand).
FLUX: add only after N_vrows_flux printed NULL_DONE ... PASS (null_vrows.py exits 0 on failure, so the queue does not retry).
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from add_stage2 import eval_pass                                              # noqa: E402
from amend_queue_e4 import last_pass                                          # noqa: E402
from make_queue import ENV, GEN_MEM, PY                                       # noqa: E402

RUN, NULL = str(HERE/'run_e06.py'), str(HERE/'null_vrows.py')
SHARDS = 4


def jobs_for(model, jobs):
    if any(j['id'] == f'Er_{model}_e06' for j in jobs):
        return []
    names = C.vrows_new(model)
    out = [dict(id=f'N_vrows_{model}', mem=GEN_MEM[model], prio=3,
                cmd=ENV+[PY, NULL, '--model', model, '--gpu', '{gpu}'])]
    wait = [f'N_vrows_{model}']+([j['id'] for j in jobs if j['id'].startswith(('W_flux_s', 'U_flux_s'))] if model == 'flux' else [])
    gen = [f'VROW_{model}_s{i}' for i in range(SHARDS)]
    out += [dict(id=g, mem=GEN_MEM[model], prio=13, after=wait,
                 cmd=ENV+[PY, RUN, '--model', model, '--split', 'e06', '--modes', *names, '--shard', f'{i}/{SHARDS}',
                          '--gpu', '{gpu}']) for i, g in enumerate(gen)]
    return out+eval_pass(model, 'r', gen, prev=last_pass(jobs, model), modes=names)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('models', nargs='+')
    a.add_argument('--null-only', action='store_true', help='add only the null jobs')
    x = a.parse_args()
    q = C.OUT/'queue'
    jobs = json.loads((q/'jobs.json').read_text())
    have = {j['id'] for j in jobs}
    new = [j for m in x.models for j in jobs_for(m, jobs) if j['id'] not in have]
    if x.null_only:
        new = [j for j in new if j['id'].startswith('N_vrows_')]
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs+new, indent=1))
    tmp.replace(q/'jobs.json')
    print(len(new), 'jobs added:', [(j['id'], j.get('after')) for j in new])


if __name__ == '__main__':
    main()
