"""PROTOCOL_ko.md §20 (E4 guard): null checks for the combination (image V + pad V, pad 1/4) with the E3 blocks.
One prompt (comp_03), seeds 71000-71001. Must be pixel-identical:
  - combination at lam 1 and pad factor 1 + both blocks vs Original;
  - the probe with the combination at lam 2.01 (no block) vs the package Controller (the unguarded upper point is made by
    run_e06.py with the Controller, the guarded points by the probe).
Must differ (the twin pass is unedited, so the block acts): lam 2.01 + pad 1/4 + each block vs lam 2.01 + pad 1/4.
  python experiments/E06_boundary/null_e4.py --model flux --gpu 0     -> <model>/null_checks_e4.json (exit 1 on failure)
"""
import argparse
import json
from pathlib import Path
import sys
import time
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from div import config, models                                                # noqa: E402
from div.method import Controller                                             # noqa: E402
import common as C                                                            # noqa: E402
from probe import Probe                                                       # noqa: E402
from run_probe import RES, _diff                                              # noqa: E402


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=['sd35', 'flux'])
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--pid', default='comp_03')
    x = a.parse_args()
    model = x.model
    hi = C.e4_names(model)[0][0]                               # e.g. v_l2.01+padv0.25
    p, o, _ = C.all_modes(model)[hi]
    row = next(r for r in C.prompts('e06') if r['id'] == x.pid)
    seeds = C.SIGNAL_SEEDS
    cfg0 = config.load(model, 'original', RES)
    cfg1 = config.load(model, p, RES+list(o)+['v_edit.lam=1.0', 'pad_v.factor=1.0'])
    cfgh = config.load(model, p, RES+list(o))
    pipe = models.load(cfg0.model, x.gpu)
    emb = models.encode(pipe, cfg0.model, row['prompt'])
    run = lambda cfg, ctrl: models.generate(pipe, cfg.model, emb, seeds, ctrl)
    ref, href = run(cfg0, None), run(cfgh, Controller(pipe, cfgh, row['prompt'], seeds))
    out = dict(prompt=row['prompt'], seeds=seeds, point=hi, v_edit=dict(cfgh.method.v_edit), pad_v=dict(cfgh.method.pad_v))
    cases = dict(combo1_fixboth=(cfg1, ('it', 'txt'), ref, 'same'), combo_probe_nofix=(cfgh, (), href, 'same'),
                 combo_fixit=(cfgh, ('it',), href, 'differ'), combo_fixtxt=(cfgh, ('txt',), href, 'differ'))
    bad = []
    for name, (cfg, twin, against, want) in cases.items():
        t0 = time.monotonic()
        ctrl = Probe(pipe, cfg, row['prompt'], seeds, twin=twin)
        imgs = run(cfg, ctrl)
        d = [_diff(a_, b_) for a_, b_ in zip(imgs, against)]
        same = not any(v['max'] for v in d)
        out[name] = dict(want=want, diff=d, twin_steps=sorted(ctrl.twin_steps), seconds=time.monotonic()-t0)
        if same != (want == 'same'):
            bad.append(name)
        print('NULL', name, json.dumps(out[name])[:400], flush=True)
    dst = C.OUT/model/'null_checks_e4.json'
    dst.write_text(json.dumps(out, indent=1, default=str))
    print('NULL_DONE', dst, 'FAILED' if bad else 'PASS', bad, flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
