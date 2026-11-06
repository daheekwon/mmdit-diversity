"""GPU check of the time-mean V edit (PROTOCOL_ko.md section 7c) before the 16 x 8 runs.

  python experiments/E02_t2v_cogvideox_pilot/check_time_mean.py --gpu 0

Short videos (9 frames = 3 latent frames), the first tune prompt, one seed, presets ours and ours_time_mean.
In every edited attention call of ours_time_mean it measures, on the returned V rows (float16):
  frame_variation  ||added - its mean over frames|| / ||added||, added = V' - V          expected ~0 (rounding only)
  mean_error       ||mean over frames of added - mean over frames of the plain edit|| / ||the latter||   expected ~0
  plain_variation  the first quantity for the plain edit (lam - 1) Q Q^T x               for comparison (7b: ~0.8)
The controller's call counts are verified by models.generate. Results: outputs/cogvideox/pilot/time_mean_check.json.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from div import config, data, models                                          # noqa: E402
from div.method import Controller                                             # noqa: E402

OUT = config.ROOT/'outputs'/'cogvideox'/'pilot'


def rel(a, b):
    return float(a.norm()/b.norm())


class Check(Controller):
    rows = []

    def image_value(self, block, v, bias, act):
        out = super().image_value(block, v, bias, act)
        if 'v_edit' in act:
            B, n, C = v.shape
            f, P = n//self.frame_tokens, self.frame_tokens
            add = (out.float()-v.float()).view(B, f, P, C)
            Q = self._q(block, v.device)
            plain = ((self.lam-1)*(((v.float()-bias.float())@Q)@Q.T)).view(B, f, P, C)
            self.rows.append(dict(step=self.step, block=block,
                                  frame_variation=rel(add-add.mean(1, keepdim=True), add),
                                  mean_error=rel(add.mean(1)-plain.mean(1), plain.mean(1)),
                                  plain_variation=rel(plain-plain.mean(1, keepdim=True), plain)))
        return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--gpu', type=int, default=0)
    x = a.parse_args()
    over = ['model.num_frames=9']
    cfg = {p: config.load('cogvideox', p, over) for p in ('ours', 'ours_time_mean')}
    r = data.prompts(cfg['ours'].data, 'vbench_tune', 16, 8)[0]
    seeds = r['seeds'][:1]
    pipe = models.load(cfg['ours'].model, x.gpu)
    emb = models.encode(pipe, cfg['ours'].model, r['prompt'])
    plain = models.generate(pipe, cfg['ours'].model, emb, seeds, Controller(pipe, cfg['ours'], r['prompt'], seeds))
    ctrl = Check(pipe, cfg['ours_time_mean'], r['prompt'], seeds)
    tm = models.generate(pipe, cfg['ours_time_mean'].model, emb, seeds, ctrl)
    rows = Check.rows
    d = np.abs(np.stack(plain).astype(np.int16)-np.stack(tm).astype(np.int16))
    res = dict(prompt=r['id'], seed=seeds[0], frames=9, frame_tokens=ctrl.frame_tokens, lam=ctrl.lam, calls=len(rows),
               frame_variation_max=max(c['frame_variation'] for c in rows),
               mean_error_max=max(c['mean_error'] for c in rows),
               plain_variation_mean=float(np.mean([c['plain_variation'] for c in rows])),
               pixels_ours_vs_time_mean=dict(mean_abs=float(d.mean()), max_abs=int(d.max())), info=ctrl.info())
    (OUT/'time_mean_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
