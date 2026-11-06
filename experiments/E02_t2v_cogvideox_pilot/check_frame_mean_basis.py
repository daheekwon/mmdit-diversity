"""GPU check of fit_basis_frame_mean.FrameMeanMoments on the real model before the calibration capture (section 7d).

  python experiments/E02_t2v_cogvideox_pilot/check_frame_mean_basis.py --gpu 0

The first tune prompt (vd048) with its 8 generation seeds, steps 0-1, all 30 blocks, 49 frames, Original model. One run
feeds the same V rows to FrameMeanMoments and to the section-7b diagnostic (temporal_share.TemporalShare). Per block:
  tr D (raw sum over both CFG groups and both steps)          vs  sum of (d_all - d_tv) / 13      expected equal
  tr(Q^T D Q), Q = the token basis bases/cogvideox_pilot_k256  vs  sum of (dq_all - dq_tv) / 13    expected equal
and, for information, the same diagnostic sums stored in outputs/cogvideox/pilot/temporal_share.json (the full run).
Results: outputs/cogvideox/pilot/frame_mean_basis_check.json.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from div import config, data, models                                          # noqa: E402
from div.method import Controller, load_basis                                 # noqa: E402
from fit_basis_frame_mean import FrameMeanMoments                             # noqa: E402
from temporal_share import TemporalShare                                      # noqa: E402

OUT = config.ROOT/'outputs'/'cogvideox'/'pilot'
STEPS = [0, 1]


class Tee:
    def __init__(self, *parts):
        self.parts, self.blocks, self.steps = parts, parts[0].blocks, parts[0].steps

    def add(self, block, x):
        for p in self.parts:
            p.add(block, x)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--gpu', type=int, default=0)
    x = a.parse_args()
    cfg = config.load('cogvideox', 'original')
    r = data.prompts(cfg.data, 'vbench_tune', 16, 8)[0]
    blocks = config.parse_range('all', models.N_BLOCKS['cogvideox'])
    Q = load_basis(str(config.resolve('bases/cogvideox_pilot_k256.pt')))['Q']
    pipe = models.load(cfg.model, x.gpu)
    fm = FrameMeanMoments(blocks, STEPS, len(r['seeds']), models.frame_tokens(cfg.model))
    ts = TemporalShare(Q, STEPS, pipe._execution_device)
    ctrl = Controller(pipe, cfg, r['prompt'], r['seeds'], capture=Tee(fm, ts))
    ts.ctrl = ctrl
    models.generate(pipe, cfg.model, models.encode(pipe, cfg.model, r['prompt']), r['seeds'], ctrl, last_step=max(STEPS))
    stored = [row for row in json.loads((OUT/'temporal_share.json').read_text())[r['id']] if row['step'] in STEPS]
    F = 13
    res, worst = {}, 0.
    for b in blocks:
        D = fm.acc[(0, b)]['D']
        q = Q[b].double()
        mine = [row for row in ts.rows if row['block'] == b]
        old = [row for row in stored if row['block'] == b]
        e = dict(trD=float(torch.trace(D)), diag=sum(c['d_all']-c['d_tv'] for c in mine)/F,
                 diag_stored=sum(c['d_all']-c['d_tv'] for c in old)/F,
                 trQDQ=float(torch.trace(q.T@D@q)), diag_q=sum(c['dq_all']-c['dq_tv'] for c in mine)/F,
                 diag_q_stored=sum(c['dq_all']-c['dq_tv'] for c in old)/F)
        e['rel_D'] = abs(e['trD']-e['diag'])/e['diag']
        e['rel_QDQ'] = abs(e['trQDQ']-e['diag_q'])/e['diag_q']
        e['rel_stored'] = abs(e['diag']-e['diag_stored'])/e['diag']
        worst = max(worst, e['rel_D'], e['rel_QDQ'])
        res[b] = e
    out = dict(prompt=r['id'], seeds=r['seeds'], steps=STEPS, max_rel_error=worst,
               max_rel_vs_stored=max(e['rel_stored'] for e in res.values()), blocks=res)
    (OUT/'frame_mean_basis_check.json').write_text(json.dumps(out, indent=1))
    print(f"max relative error {worst:.2e}, vs stored diagnostic {out['max_rel_vs_stored']:.2e}")


if __name__ == '__main__':
    main()
