"""DESIGN_vrows_ko.md §5: null checks of v_edit.rows=image against the stored e06 images (same prompt, the 8 e06 seeds in
one batch of 8, as run_e06.py made them). Must be pixel-identical:
  all_v      rows=all at the final lam, V edit alone   vs  stored v_l<lam>          (the code change leaves rows=all as it was)
  all_combo  rows=all at the final lam + pad 1/4      vs  stored final combination
  img_lam1   rows=image at lam 1                       vs  stored original
  img_lam1p  rows=image at lam 1 + pad 1/4            vs  stored padv0.25
Must differ: img_v = rows=image at the final lam vs stored v_l<lam>.
  python experiments/E06_boundary/null_vrows.py --model sd35 --gpu 0    -> <model>/null_checks_vrows.json ('pass')
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from div import config, models                                                # noqa: E402
from div.method import Controller                                             # noqa: E402
import common as C                                                            # noqa: E402

FINAL = dict(sd35='1.14', flux='1.72')


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=['sd35', 'flux'])
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--pid', default='comp_03')
    x = a.parse_args()
    m, lam = x.model, FINAL[x.model]
    res = [f'model.resolution={C.RES}']
    row = next(r for r in C.prompts('e06') if r['id'] == x.pid)
    gen = C.gen_root(m, 'e06')/x.pid
    seeds = C.SEEDS['e06']
    base = config.load(m, 'original', res)
    pipe = models.load(base.model, x.gpu)
    emb = models.encode(pipe, base.model, row['prompt'])
    stored = lambda mode: [np.asarray(Image.open(gen/mode/f'seed{s}.png').convert('RGB'), dtype=np.int16) for s in seeds]
    final = f'v_l{lam}+padv0.25'
    cases = dict(all_v=('v_edit', [f'v_edit.lam={lam}'], f'v_l{lam}', 'same'),
                 all_combo=('ours', C.all_modes(m)[final][1], final, 'same'),
                 img_lam1=('v_edit', ['v_edit.lam=1.0', 'v_edit.rows=image'], C.ORIG[m], 'same'),
                 img_lam1p=('ours', ['v_edit.lam=1.0', 'v_edit.rows=image'], 'padv0.25', 'same'),
                 img_v=('v_edit', [f'v_edit.lam={lam}', 'v_edit.rows=image'], f'v_l{lam}', 'differ'))
    out, bad = dict(prompt=row['prompt'], seeds=seeds), []
    for name, (preset, over, ref, want) in cases.items():
        cfg = config.load(m, preset, res+list(over))
        t0 = time.monotonic()
        ctrl = Controller(pipe, cfg, row['prompt'], seeds)
        imgs = [np.asarray(im.convert('RGB'), dtype=np.int16) for im in models.generate(pipe, cfg.model, emb, seeds, ctrl)]
        d = [np.abs(i-r) for i, r in zip(imgs, stored(ref))]
        mx = int(max(z.max() for z in d))
        same = mx == 0
        out[name] = dict(want=want, against=ref, max=mx, mean=float(np.mean([z.mean() for z in d])),
                         rows=ctrl.v_rows, seconds=time.monotonic()-t0)
        if same != (want == 'same'):
            bad.append(name)
        print('NULL', name, json.dumps(out[name]), flush=True)
    out['pass'] = not bad
    dst = C.OUT/m/'null_checks_vrows.json'
    dst.write_text(json.dumps(out, indent=1))
    print('NULL_DONE', dst, 'FAILED' if bad else 'PASS', bad, flush=True)
    # exit 0 either way: a deterministic failure must not be retried by the queue; the generation jobs are added
    # (amend_queue_vrows.py) only after a PASS


if __name__ == '__main__':
    main()
