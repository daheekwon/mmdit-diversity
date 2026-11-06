"""Diagnostic of PROTOCOL_ko.md section 7b: the share of the V edit's added value (Q Q^T x) and of the seed deviation
that varies across the latent frames, per step, block and CFG branch. Original model, steps 0-20, the 16 tune prompts
with the seeds of the generated videos, basis bases/cogvideox_pilot_k256.pt.

  python experiments/E02_t2v_cogvideox_pilot/temporal_share.py --gpu 0
  python experiments/E02_t2v_cogvideox_pilot/temporal_share.py --gpu 0 --test      # 9 frames, 1 prompt, 2 seeds, steps 0-1

Video tokens are frame-major, so the bias-free V rows of one (prompt, step, block, branch) are x [seeds, frames,
positions 1350, C]. With a = x Q (Q orthonormal, k 256) and tv(t) = ||t - mean over frames of t||^2:
  edit_all = ||a||^2, edit_tv = tv(a)       the added value Q Q^T x (up to lam - 1)
  x_all = ||x||^2,    x_tv = tv(x)          whole space; outside Q = whole - edit
  dq_all, dq_tv / d_all, d_tv               the same for the seed deviation x - mean over seeds of x (in Q / whole)
Sums are saved after every prompt in outputs/cogvideox/pilot/temporal_share.json (resumable).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from div import config, data, models                                          # noqa: E402
from div.method import Controller, load_basis                                 # noqa: E402

OUT = config.ROOT/'outputs'/'cogvideox'/'pilot'
POSITIONS = 1350                                                              # 30 x 45 patches at 480 x 720
BRANCHES = ('uncond', 'cond')                                                 # CFG batch order of the pipeline


def sq(t):
    return float(t.pow(2).sum(dtype=torch.float64))


def tv(t):
    return sq(t-t.mean(1, keepdim=True))


class TemporalShare:
    def __init__(self, Q, steps, device):
        self.blocks, self.steps = sorted(Q), list(steps)
        self.Q = {b: q.to(device, torch.float32) for b, q in Q.items()}
        self.ctrl, self.rows = None, []

    @torch.no_grad()
    def add(self, block, x):
        n = len(self.ctrl.seeds)
        if x.shape[0] != 2*n or x.shape[1] % POSITIONS:
            raise ValueError(f'unexpected V rows {tuple(x.shape)}')
        x = x.view(x.shape[0], x.shape[1]//POSITIONS, POSITIONS, -1)
        for g, branch in enumerate(BRANCHES):
            X = x[g*n:(g+1)*n]
            a = X@self.Q[block]
            row = dict(step=self.ctrl.step, block=block, branch=branch, edit_all=sq(a), edit_tv=tv(a))
            row.update(x_all=sq(X), x_tv=tv(X))
            a = a-a.mean(0)
            row.update(dq_all=sq(a), dq_tv=tv(a))
            del a
            d = X-X.mean(0)
            row.update(d_all=sq(d), d_tv=tv(d))
            self.rows.append(row)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--test', action='store_true')
    x = a.parse_args()
    cfg = config.load('cogvideox', 'original', ['model.num_frames=9'] if x.test else [])
    rows = data.prompts(cfg.data, 'vbench_tune', 16, 8)
    steps = config.parse_range(cfg.method.v_edit.steps, cfg.model.steps)
    if x.test:
        rows, steps = [dict(rows[0], seeds=rows[0]['seeds'][:2])], steps[:2]
    Q = load_basis(str(config.resolve(cfg.method.v_edit.basis)))['Q']
    out = OUT/('temporal_share_test.json' if x.test else 'temporal_share.json')
    res = json.loads(out.read_text()) if out.exists() else {}
    pipe = models.load(cfg.model, x.gpu)
    for r in rows:
        if r['id'] in res:
            continue
        t0 = time.monotonic()
        cap = TemporalShare(Q, steps, pipe._execution_device)
        ctrl = Controller(pipe, cfg, r['prompt'], r['seeds'], capture=cap)
        cap.ctrl = ctrl
        models.generate(pipe, cfg.model, models.encode(pipe, cfg.model, r['prompt']), r['seeds'], ctrl, last_step=max(steps))
        res[r['id']] = cap.rows
        tmp = out.with_name(f'{out.stem}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(res))
        os.replace(tmp, out)
        e = sum(c['edit_tv'] for c in cap.rows)/sum(c['edit_all'] for c in cap.rows)
        print(f"{r['id']} {len(cap.rows)} rows, edit temporal share {e:.3f}, {time.monotonic()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
