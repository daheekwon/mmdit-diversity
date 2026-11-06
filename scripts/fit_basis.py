"""Fit the V-edit basis (div/basis.py) on calibration prompts with the Original model.

  python scripts/fit_basis.py --model sd35 --name sd35_calib_k256 --prompts calib.txt [--blocks 6-23] [--steps 0-3]
  python scripts/fit_basis.py --model sd35 --name sd35_calib_k64 --refit sd35_calib_k256 --k 64     # no GPU
  python scripts/fit_basis.py --model flux --legacy                     # old dev-8 basis (SD3.5 6-23, FLUX 19-56)
  python scripts/fit_basis.py --model sd35 --legacy --blocks all        # old dev-8 basis, SD3.5 blocks 0-23 (default)
  python scripts/fit_basis.py --model klein --name klein_dev8 --dataset dev8 --steps 0-4 0-6   # one capture, two bases

Writes bases/<name>.pt (Q, unit per block, held-out selectivity) and bases/<name>.json (report); the capture moments
go to bases/<name>_moments.pt so that k / beta can be refit without the GPU. Defaults: all blocks (so v_edit.blocks
can change without a refit), the config's v_edit steps, calibration prompts from data.calibration.prompts, seeds
data.calibration.seed_base + 0..N-1. --dataset takes the prompts and seeds of a data set instead (dev8: the 8 dev prompts
with seeds 40000-40015, the calibration of every earlier basis). Several --steps sets share one capture run and give one
basis each, bases/<name>_s<set>_k<k>.pt (div/basis.StepSets; each equals a separate capture over its steps).
"""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import basis, config, data, models                                   # noqa: E402

BASES = config.ROOT/'bases'


def save(b, name):
    torch.save(b, BASES/f'{name}.pt')
    rep = {k: v for k, v in b.items() if k != 'Q'}
    (BASES/f'{name}.json').write_text(json.dumps(rep, indent=1, default=str))
    sel = [c['selectivity_ab'] for c in b['check'].values()]+[c['selectivity_ba'] for c in b['check'].values()]
    print(f"saved bases/{name}.pt: blocks {b['blocks'][0]}-{b['blocks'][-1]}, unit {min(b['unit'].values()):.3f}-"
          f"{max(b['unit'].values()):.3f}" + (f', held-out selectivity mean {sum(sel)/len(sel):.2f} min {min(sel):.2f}' if sel else ''))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=['sd35', 'flux', 'cogvideox', 'klein'])
    a.add_argument('--name')
    a.add_argument('--legacy', action='store_true')
    a.add_argument('--refit', help='name of an earlier fit whose moments to reuse')
    a.add_argument('--prompts', help='calibration prompt file (txt / json / jsonl)')
    a.add_argument('--seeds', type=int)
    a.add_argument('--dataset', help='calibration prompts and seeds from this data set (e.g. dev8)')
    a.add_argument('--blocks', help='default all')
    a.add_argument('--steps', nargs='+')
    a.add_argument('--k', type=int, default=256)
    a.add_argument('--beta', type=float, default=1.)
    a.add_argument('--gpu', type=int, default=0)
    x = a.parse_args()
    if x.legacy:
        whole = x.blocks == 'all'
        return save(basis.legacy(x.model, whole), x.name or f'{x.model}_legacy_dev8{"_b0to23" if whole else ""}_k256')
    if not x.name:
        a.error('--name is required')
    cfg = config.load(x.model, 'original')
    if x.steps and len(x.steps) > 1:
        if x.refit:
            a.error('--refit takes one step set')
        return fit_sets(x, cfg)
    if x.steps:
        x.steps = x.steps[0]
    if x.refit:
        mom = torch.load(BASES/f'{x.refit}_moments.pt', weights_only=True)
    else:
        if x.dataset:
            rows = data.prompts(cfg.data, x.dataset)
            prompts, seeds = [r['prompt'] for r in rows], rows[0]['seeds'][:x.seeds or len(rows[0]['seeds'])]
        else:
            if x.prompts:
                cfg.data.calibration.prompts = x.prompts
            prompts = data.calibration_prompts(cfg.data)
            seeds = [cfg.data.calibration.seed_base+i for i in range(x.seeds or cfg.data.calibration.seeds)]
        nb = models.N_BLOCKS[x.model]
        blocks = config.parse_range(x.blocks or 'all', nb)          # all blocks, so v_edit.blocks can change later
        steps = config.parse_range(x.steps or cfg.method.v_edit.steps, cfg.model.steps)
        pipe = models.load(cfg.model, x.gpu)
        state = basis.capture(pipe, cfg, prompts, seeds, blocks, steps)
        bias = biases(pipe, x.model, blocks)
        mom = dict(state=state, bias=bias, prompts=prompts, seeds=seeds, steps=steps)
        torch.save(mom, BASES/f'{x.name}_moments.pt')
    meta = dict(prompts=mom['prompts'], seeds=mom['seeds'], steps=mom['steps'])
    save(basis.fit(mom['state'], mom['bias'], x.model, x.k, x.beta, meta), x.name)


def biases(pipe, model, blocks):
    """to_v bias per block (zeros where the model has none: FLUX.2)."""
    out = {}
    for b in blocks:
        attn = models.attentions(pipe, model)[b]
        bias = models.value_bias(attn)
        out[b] = torch.zeros(models.value_dim(attn)) if bias is None else bias.detach().float().cpu()
    return out


def fit_sets(x, cfg):
    """Several step sets from one capture run (basis.capture_sets)."""
    if x.dataset:
        rows = data.prompts(cfg.data, x.dataset)
        prompts, seeds = [r['prompt'] for r in rows], rows[0]['seeds'][:x.seeds or len(rows[0]['seeds'])]
    else:
        if x.prompts:
            cfg.data.calibration.prompts = x.prompts
        prompts = data.calibration_prompts(cfg.data)
        seeds = [cfg.data.calibration.seed_base+i for i in range(x.seeds or cfg.data.calibration.seeds)]
    blocks = config.parse_range(x.blocks or 'all', models.N_BLOCKS[x.model])
    sets = {f's{r}': config.parse_range(r, cfg.model.steps) for r in x.steps}
    pipe = models.load(cfg.model, x.gpu)
    states = basis.capture_sets(pipe, cfg, prompts, seeds, blocks, sets)
    bias = biases(pipe, x.model, blocks)
    for name, state in states.items():
        full = f'{x.name}_{name}_k{x.k}'
        torch.save(dict(state=state, bias=bias, prompts=prompts, seeds=seeds, steps=sets[name]), BASES/f'{full}_moments.pt')
        save(basis.fit(state, bias, x.model, x.k, x.beta, dict(prompts=prompts, seeds=seeds, steps=sets[name])), full)


if __name__ == '__main__':
    main()
