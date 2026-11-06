"""Stage 1 queue of E06 (PROTOCOL_ko.md §3, §8): links the ho4 Original / DAVE folders, then writes
outputs/e06_boundary/queue/jobs.json. Start the queue with
  nohup python experiments/E01_res1024_coco48/job_queue.py outputs/e06_boundary/queue > outputs/e06_boundary/queue/nohup.out &
Re-running rewrites jobs.json with the stage-1 ids only: it drops stage-2 jobs (add_stage2.py) and requeued copies
(<id>b). Once the queue runs, add jobs with add_stage2.py instead.
"""
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402

PY = '/DATA0/dahee/diversity/.miniconda3/envs/diversity/bin/python'
ENV = ['env', 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True']
GEN_MEM = dict(sd35=10, flux=26)
SHARDS = dict(e06=8, ho4=4)
PRIO = {('sd35', 'e06'): 10, ('flux', 'e06'): 11, ('sd35', 'ho4'): 20, ('flux', 'ho4'): 21}


def link_ho4(model):
    src = C.REPO/'value_diversity'/'results'/'generalize'/model/'test'/'generation'
    dst = C.gen_root(model, 'ho4')
    n = 0
    for r in C.prompts('ho4'):
        for mode in C.HO4_LINK[model]:
            s, d = src/r['id']/mode, dst/r['id']/mode
            if not (s/'meta.json').exists():
                raise SystemExit(f'missing {s}')
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                d.symlink_to(s, target_is_directory=True)
                n += 1
    return n


def jobs():
    out = []
    for (model, split), prio in PRIO.items():
        root = C.gen_root(model, split)
        k = SHARDS[split]
        gen = [f'G_{model}_{split}_s{i}' for i in range(k)]
        out += [dict(id=g, mem=GEN_MEM[model], prio=prio,
                     cmd=ENV+[PY, str(HERE/'run_e06.py'), '--model', model, '--split', split, '--shard', f'{i}/{k}',
                              '--gpu', '{gpu}']) for i, g in enumerate(gen)]
        ref = C.ORIG[model] if split == 'e06' else C.HO4_REF[model]
        ev = f'E_{model}_{split}'
        out.append(dict(id=ev, mem=6, prio=5, after=gen, cmd=[PY, str(C.REPO/'value_diversity'/'study_eval.py'), str(root),
                                                             '--name', 'combo', '--ref', ref, '--device', 'cuda:{gpu}']))
        out.append(dict(id=f'V_{model}_{split}', mem=0, prio=5, after=[ev],
                        cmd=[PY, '-m', 'value_diversity.vendi_eval', str(root), '--name', 'combo']))
        out += [dict(id=f'J_{model}_{split}_{j}', mem=24, prio=6, after=gen,
                     cmd=[PY, str(C.REPO/'value_diversity'/'study_judge.py'), str(root), '--name', 'e06', '--shard', f'{j}/2',
                          '--device', 'cuda:{gpu}']) for j in range(2)]
        if split == 'e06':
            out += [dict(id=f'{t}_{model}_e06', mem=m, prio=7, after=gen,
                         cmd=[PY, str(HERE/'eval_extra.py'), kind, model, '--device', 'cuda:{gpu}'])
                    for t, kind, m in (('Q', 'vqa', 24), ('O', 'objsize', 4))]
    return out


def main():
    for model in ('sd35', 'flux'):
        print(model, 'ho4 links made:', link_ho4(model))
    q = C.OUT/'queue'
    (q/'logs').mkdir(parents=True, exist_ok=True)
    js = jobs()
    (q/'jobs.json').write_text(json.dumps(js, indent=1))
    print(len(js), 'jobs ->', q/'jobs.json')


if __name__ == '__main__':
    main()
