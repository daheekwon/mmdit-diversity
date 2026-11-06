"""Mechanism measurement of PROTOCOL_ko.md section 6, at one resolution. No image is saved; each run stops early.

  Original (steps 0-2, every block): image-query softmax mass on the padding keys, mean over image queries and heads,
      and the share of image queries whose pad mass > 0.5. SD3.5 rows: cond and uncond separately.
  Our operating point (V-edit blocks and steps): realized relative image-V change sqrt(sum|V'-V|^2 / sum|V|^2)
      (V = to_v output incl. bias), next to lam and the basis unit.

  python experiments/E01_res1024_coco48/measure_mech.py --model flux --res 1024 --gpu 0
Output: outputs/res<res>/<model>/coco48/mech.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from div import config, models                                                # noqa: E402
from div.config import parse_range                                            # noqa: E402
from div.method import Controller, basis_unit, load_basis                     # noqa: E402
from run_coco48 import coco48                                                 # noqa: E402

CHUNK = 512
PAD_STEPS = range(3)


class Scope:
    def __init__(self, blocks, steps):
        self.blocks, self.steps = list(blocks), list(steps)

    def add(self, block, v):
        pass


class MechController(Controller):
    def __init__(self, pipe, cfg, prompt, seeds, capture, cfg_rows):
        super().__init__(pipe, cfg, prompt, seeds, capture=capture)
        self.pad_index = models.pad_index(pipe, self.model, prompt)
        self.cfg_rows, self.mass, self.vchange = cfg_rows, {}, {}

    @torch.no_grad()
    def image_value(self, block, v, bias, act):
        out = super().image_value(block, v, bias, act)
        if 'v_edit' in act:
            d, x = (out.float()-v.float()), v.float()
            self.vchange.setdefault((self.step, block), []).append((d.square().sum().item(), x.square().sum().item()))
        return out

    @torch.no_grad()
    def attend(self, q, k, v, img, txt, act):
        if 'capture' in act and self.step in PAD_STEPS:
            pos = (txt.start+self.pad_index).to(q.device)
            qi, kt, scale = q[:, :, img], k.float().transpose(-1, -2), 1/math.sqrt(q.shape[-1])
            mass = torch.cat([(qi[:, :, c:c+CHUNK].float()@kt*scale).softmax(-1)[..., pos].sum(-1)
                              for c in range(0, qi.shape[2], CHUNK)], 2)
            per_query = mass.mean(1)                                          # [B, n_img], mean over heads
            halves = dict(uncond=per_query[:len(per_query)//2], cond=per_query[len(per_query)//2:]) if self.cfg_rows \
                else dict(cond=per_query)
            self.mass[(self.step, self.calls[-1][1])] = {h: dict(mean=t.mean().item(), share_gt_half=(t > .5).float().mean().item())
                                                         for h, t in halves.items()}
        return super().attend(q, k, v, img, txt, act)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=['sd35', 'flux'])
    a.add_argument('--res', type=int, required=True)
    a.add_argument('--pids', nargs='*', default=[f'coco_{i:02d}' for i in range(8)])
    a.add_argument('--seeds', nargs='*', type=int, default=[68000, 68001])
    a.add_argument('--gpu', type=int, default=0)
    x = a.parse_args()
    over = [f'model.resolution={x.res}']
    orig, ours = config.load(x.model, 'original', over), config.load(x.model, 'ours', over)
    nb, steps = models.N_BLOCKS[x.model], orig.model.steps
    vb, vs = parse_range(ours.method.v_edit.blocks, nb), parse_range(ours.method.v_edit.steps, steps)
    pipe = models.load(orig.model, x.gpu)
    rows = [r for r in coco48() if r['id'] in x.pids]
    out = dict(model=x.model, res=x.res, seeds=x.seeds, pids=x.pids, pad_mass={}, v_change={})
    for r in rows:
        emb = models.encode(pipe, orig.model, r['prompt'])
        ctrl = MechController(pipe, orig, r['prompt'], x.seeds, Scope(range(nb), PAD_STEPS), x.model == 'sd35')
        models.generate(pipe, orig.model, emb, x.seeds, ctrl, last_step=max(PAD_STEPS))
        out['pad_mass'][r['id']] = {f'{s}/{b}': m for (s, b), m in ctrl.mass.items()}
        ctrl = MechController(pipe, ours, r['prompt'], x.seeds, Scope([], []), x.model == 'sd35')
        models.generate(pipe, ours.model, emb, x.seeds, ctrl, last_step=max(vs))
        out['v_change'][r['id']] = {f'{s}/{b}': math.sqrt(sum(d for d, _ in t)/sum(e for _, e in t)) for (s, b), t in ctrl.vchange.items()}
        out['lam'], out['unit'] = ctrl.lam, basis_unit(load_basis(ours.method.v_edit.basis), vb)
        pm, vc = out['pad_mass'][r['id']].values(), out['v_change'][r['id']].values()
        print(r['id'], f"pad mass (cond, cell mean) {sum(m['cond']['mean'] for m in pm)/len(pm):.4f}",
              f'v change (cell mean) {sum(vc)/len(vc):.4f}', flush=True)
    path = HERE.parents[1]/'outputs'/f'res{x.res}'/x.model/'coco48'/'mech.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print('wrote', path)


if __name__ == '__main__':
    main()
