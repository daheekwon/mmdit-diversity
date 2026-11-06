"""Frame-mean V-edit basis (PROTOCOL_ko.md section 7d): the basis of div/basis.py fitted on the per-video mean over the
13 latent frames of the bias-free video V, instead of on every video token.

  python experiments/E02_t2v_cogvideox_pilot/fit_basis_frame_mean.py --gpu 0
  python experiments/E02_t2v_cogvideox_pilot/fit_basis_frame_mean.py --report        # no GPU: section-7d report only

Video V rows are frame-major [frames x positions] (models.frame_tokens = 1350 positions per latent frame). For each
(prompt, step, CFG branch) group and block, FrameMeanMoments averages x over the frames at each position,
xbar [seeds, 1350, C], and accumulates the moments of basis.Moments on xbar:
  D = seed deviation of xbar (the part of the seed deviation that is common to all frames of a video),
  C = seed mean of xbar.
Q = top-k eigenvectors of D/tr D - C/tr C, as for the token basis; held-out selectivity on the two prompt halves.
Same calibration as the token basis bases/cogvideox_pilot_k256.pt (calib_dimension_longer_32.txt x seeds
9000000-9000007, steps 0-20, all blocks), resumable (checkpoint after every prompt).
Outputs: bases/cogvideox_pilot_framemean_k256{.pt, .json, _moments.pt}, and bases/cogvideox_pilot_framemean_k256_report.json:
per block, for the new Q and the token Q, the share of the frame-mean D and C in Q, and the relative change of the
whole video V per unit (lam - 1) of the time-mean edit, sqrt(tr(Q^T M_fm Q) / E||V||^2) with E||V||^2 per video token
from the token basis moments (the 'unit' in the .pt is the change of the frame-mean V, not of V).
"""
import argparse
import json
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
from div import basis, config, data, models                                   # noqa: E402
from fit_basis import BASES, save                                             # noqa: E402
from fit_basis_resumable import capture_resumable                             # noqa: E402

NAME, TOKEN = 'cogvideox_pilot_framemean_k256', 'cogvideox_pilot_k256'
PROMPTS = '/DATA0/dahee/data/datasets/vbench/calib_dimension_longer_32.txt'


class FrameMeanMoments(basis.Moments):
    def __init__(self, blocks, steps, n_seeds, per_frame):
        super().__init__(blocks, steps, n_seeds)
        self.per_frame = per_frame

    def add(self, block, x):
        """x: bias-free video V rows [B, frames x per_frame, C] (frame-major)."""
        B, n, C = x.shape
        if n % self.per_frame:
            raise ValueError(f'{n} video rows are not whole frames of {self.per_frame}')
        super().add(block, x.reshape(B, n//self.per_frame, self.per_frame, C).mean(1))


def report(fm, tok, bias):
    """fm, tok: moment files (dict with 'state') of the frame-mean and token captures; returns the section-7d table."""
    Qn = torch.load(BASES/f'{NAME}.pt', weights_only=True)['Q']
    Qt = torch.load(BASES/f'{TOKEN}.pt', weights_only=True)['Q']
    rows = {}
    for b in sorted(Qn):
        s, t = basis._pool(fm['state'], b), basis._pool(tok['state'], b)
        bb = bias[b].double()
        ev = torch.trace(t['D']+t['C'])+2*bb@t['H']+bb@bb                     # E||V||^2 per video token
        M = s['D']+s['C']
        r = {}
        for name, Q in (('framemean', Qn[b].double()), ('token', Qt[b].double())):
            r[name] = dict(share_D_fm=basis._share(Q, s['D']), share_C_fm=basis._share(Q, s['C']),
                           share_D_token=basis._share(Q, t['D']),
                           unit_time_mean_edit=float((torch.trace(Q.T@M@Q)/ev).sqrt()))
        rows[b] = r
    first = next(iter(rows.values()))
    pooled = {name: {k: sum(rows[b][name][k] for b in rows)/len(rows) for k in first[name]}
              for name in ('framemean', 'token')}
    return dict(blocks=rows, mean_over_blocks=pooled)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--k', type=int, default=256)
    a.add_argument('--report', action='store_true', help='only (re)write the report from saved moments')
    x = a.parse_args()
    cfg = config.load('cogvideox', 'original')
    blocks = config.parse_range('all', models.N_BLOCKS['cogvideox'])
    if not x.report:
        cfg.data.calibration.prompts = PROMPTS
        prompts = data.calibration_prompts(cfg.data)
        seeds = [cfg.data.calibration.seed_base+i for i in range(8)]
        steps = config.parse_range(cfg.method.v_edit.steps, cfg.model.steps)
        pipe = models.load(cfg.model, x.gpu)
        mom = FrameMeanMoments(blocks, steps, len(seeds), models.frame_tokens(cfg.model))
        ckpt = BASES/f'{NAME}_capture_ckpt.pt'
        state = capture_resumable(pipe, cfg, prompts, seeds, blocks, steps, ckpt, mom=mom)
        bias = {b: models.attentions(pipe, 'cogvideox')[b].to_v.bias.detach().float().cpu() for b in blocks}
        torch.save(dict(state=state, bias=bias, prompts=prompts, seeds=seeds, steps=steps, frame_mean=True),
                   BASES/f'{NAME}_moments.pt')
        save(basis.fit(state, bias, 'cogvideox', x.k, 1., dict(prompts=prompts, seeds=seeds, steps=steps,
                                                                  frame_mean=True)), NAME)
        ckpt.unlink()
    fm = torch.load(BASES/f'{NAME}_moments.pt', weights_only=False)
    tok = torch.load(BASES/f'{TOKEN}_moments.pt', weights_only=False)
    rep = report(fm, tok, fm['bias'])
    (BASES/f'{NAME}_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep['mean_over_blocks'], indent=1))


if __name__ == '__main__':
    main()
