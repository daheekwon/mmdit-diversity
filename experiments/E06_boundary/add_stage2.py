"""Append the stage-2 jobs (PROTOCOL_ko.md §9) to the running E06 queue; jobs already present are left as they are.

  python experiments/E06_boundary/add_stage2.py [--e3]

Without --e3: FLUX null check and write measure, the random-direction condition (after the write measure), signal
capture (after the null check), and a second evaluation pass over e06 once those images exist.
--e3 --pad sd35:padv<f> flux:padv<f>: the E3 conditions (after the E1 knee check and the E2 onset rule) and their
evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from make_queue import ENV, GEN_MEM, PY                                       # noqa: E402

PROBE = str(HERE/'run_probe.py')
SIG_MEM = dict(sd35=12, flux=30)
E3_MEM = dict(sd35=14, flux=32)


def stage2(model):
    out, root = [], C.gen_root(model, 'e06')
    pre = []
    if model == 'flux':
        out.append(dict(id='N_flux', mem=32, prio=4, cmd=ENV+[PY, PROBE, 'null', '--model', 'flux', '--gpu', '{gpu}']))
        out.append(dict(id='W_flux', mem=32, prio=4, cmd=ENV+[PY, PROBE, 'write', '--model', 'flux', '--gpu', '{gpu}']))
        pre = ['N_flux', 'W_flux']
    rnd = [f'R_{model}_s{i}' for i in range(2)]
    out += [dict(id=g, mem=GEN_MEM[model], prio=12, after=pre,
                 cmd=ENV+[PY, str(HERE/'run_e06.py'), '--model', model, '--split', 'e06', '--modes', 'RANDOM',
                          '--shard', f'{i}/2', '--gpu', '{gpu}']) for i, g in enumerate(rnd)]
    sig = [f'S_{model}_s{i}' for i in range(4)]
    out += [dict(id=g, mem=SIG_MEM[model], prio=13, after=pre,
                 cmd=ENV+[PY, PROBE, 'signals', '--model', model, '--shard', f'{i}/4', '--gpu', '{gpu}'])
            for i, g in enumerate(sig)]
    gen1 = [f'G_{model}_e06_s{i}' for i in range(8)]
    out += eval_pass(model, '2', gen1+rnd)
    return out


def eval_pass(model, tag, after, prev='', modes=None):
    """Evaluation of the e06 root once `after` are done. Passes write the same files (eval csv / cache, judge, vqa,
    objsize csv), so each job also waits for the same job of the previous pass (prev = its tag, '' = stage 1)."""
    root = C.gen_root(model, 'e06')
    ev = f'E{tag}_{model}_e06'
    out = [dict(id=ev, mem=6, prio=5, after=after+[f'E{prev}_{model}_e06'], cmd=[PY, str(C.REPO/'value_diversity'/'study_eval.py'), str(root),
                                                        '--name', 'combo', '--ref', C.ORIG[model], '--device', 'cuda:{gpu}']),
           dict(id=f'V{tag}_{model}_e06', mem=0, prio=5, after=[ev],
                cmd=[PY, '-m', 'value_diversity.vendi_eval', str(root), '--name', 'combo'])]
    only = ['--modes', *modes] if modes else []          # judge only the new conditions (no re-judging across passes)
    out += [dict(id=f'J{tag}_{model}_e06_{j}', mem=24, prio=6, after=after+[f'J{prev}_{model}_e06_{j}'],
                 cmd=[PY, str(C.REPO/'value_diversity'/'study_judge.py'), str(root), '--name', 'e06', '--shard', f'{j}/2',
                      '--device', 'cuda:{gpu}']+only) for j in range(2)]
    out += [dict(id=f'{t}{tag}_{model}_e06', mem=m, prio=7, after=after+[f'{t}{prev}_{model}_e06'],
                 cmd=[PY, str(HERE/'eval_extra.py'), kind, model, '--device', 'cuda:{gpu}'])
            for t, kind, m in (('Q', 'vqa', 24), ('O', 'objsize', 4))]
    return out


def e3(model, pad):
    """pad: the E2 onset condition of this model (PROTOCOL §3 / §11 rule), e.g. 'padv0.5'."""
    if pad not in C.E3_PAD:
        raise SystemExit(f'--pad {model}:{pad}: not one of {C.E3_PAD}')
    names = [f'{b}+fix{x}' for b in C.E3_V[model]+[pad] for x in ('it', 'txt')]
    gen = [f'X_{model}_s{i}' for i in range(8)]
    out = [dict(id=g, mem=E3_MEM[model], prio=12, cmd=ENV+[PY, PROBE, 'e3', '--model', model, '--modes', *names,
                                                          '--shard', f'{i}/8', '--gpu', '{gpu}']) for i, g in enumerate(gen)]
    return out+eval_pass(model, '3', gen, prev='c', modes=names)   # after pass 'c' (amend_queue_c.py)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--e3', action='store_true')
    a.add_argument('--pad', nargs='*', default=[], help='with --e3: sd35:padv<f> flux:padv<f> (E2 onset rule, PROTOCOL §3)')
    x = a.parse_args()
    pad = dict(p.split(':') for p in x.pad)
    if x.e3 and not pad:
        raise SystemExit('--e3 needs --pad <model>:padv<f> for each model to add (models are decided separately)')
    q = C.OUT/'queue'
    jobs = json.loads((q/'jobs.json').read_text())
    have = {j['id'] for j in jobs}
    models = sorted(pad) if x.e3 else ('sd35', 'flux')
    new = [j for m in models for j in (e3(m, pad[m]) if x.e3 else stage2(m)) if j['id'] not in have]
    tmp = q/'jobs.json.tmp'
    tmp.write_text(json.dumps(jobs+new, indent=1))
    tmp.replace(q/'jobs.json')
    print(len(new), 'jobs added:', [j['id'] for j in new])


if __name__ == '__main__':
    main()
