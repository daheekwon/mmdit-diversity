"""GPU checks of the FLUX.2-klein adapter against the native diffusers pipeline (no earlier klein images exist).

  python tests/parity_klein.py --gpu 2 [--speed]

Prompt dev_animal_00 ('A red panda'), seeds 50000-50001 (batch 2), 512 x 512, 50 steps, CFG 4.
  native            pipe(prompt=..., num_images_per_prompt=2, generator per seed)              reference
  original          models.generate without a controller (prompt and empty prompt encoded once)  expected: same
  v_edit lam 1      every block 0-24, every step, random orthonormal basis (k 64)              expected: same
  pad_v x1          every block, every step (image-query rows through Controller.attend)       expected: same
  dave a1           block 2 (double) and block 13 (single, image rows of [text | image])        expected: same
  ours smoke        lam 1.5 on blocks 5-24 steps 0-4 + pad 1/4 steps 0-4 (random basis): images change, call counts
--speed: one batch of 16 Original images (time per image). Results: outputs/parity/klein.json.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import config, models                                                # noqa: E402
from div.method import Controller                                             # noqa: E402

PROMPT, SEEDS = 'A red panda', [50000, 50001]


def arr(images):
    return np.stack([np.asarray(im.convert('RGB'), dtype=np.int16) for im in images])


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--gpu', type=int, required=True)
    a.add_argument('--speed', action='store_true')
    a.add_argument('--scratch', default='/tmp')
    x = a.parse_args()
    base = config.load('klein', 'original')
    pipe = models.load(base.model, x.gpu)
    m = base.model
    out = {}
    t0 = time.monotonic()
    ref = pipe(prompt=PROMPT, num_images_per_prompt=len(SEEDS), height=m.resolution, width=m.resolution,
               num_inference_steps=m.steps, guidance_scale=m.guidance, max_sequence_length=m.max_sequence_length,
               generator=[torch.Generator('cpu').manual_seed(s) for s in SEEDS]).images
    out['native_seconds_batch2'] = time.monotonic()-t0
    R = arr(ref)
    emb = models.encode(pipe, m, PROMPT)

    def check(name, cfg, expect_same=True):
        ctrl = Controller(pipe, cfg, PROMPT, SEEDS) if any(cfg.method[c].enabled for c in config.COMPONENTS) else None
        t = time.monotonic()
        ims = models.generate(pipe, cfg.model, emb, SEEDS, ctrl)
        A = arr(ims)
        d = np.abs(A-R)
        out[name] = dict(same=bool((d == 0).all()), max_abs=int(d.max()), mean_abs=float(d.mean()),
                         seconds=time.monotonic()-t, info=ctrl.info() if ctrl else None,
                         expected='same' if expect_same else 'different')
        ok = out[name]['same'] == expect_same
        out[name]['pass'] = bool(ok)
        print(name, out[name], flush=True)
        return ims

    # random orthonormal basis for every block (V dim 3072), unit 1
    g = torch.Generator().manual_seed(0)
    Q = {b: torch.linalg.qr(torch.randn(3072, 64, generator=g))[0] for b in range(25)}
    bpath = Path(x.scratch)/'klein_random_k64.pt'
    torch.save(dict(model='klein', k=64, beta=1., blocks=list(range(25)), Q=Q, unit={b: 1. for b in range(25)}, check={},
                    source='random (parity test)'), bpath)
    check('original', base)
    check('v_edit_lam1_all', config.load('klein', 'v_edit', [f'v_edit.basis={bpath}', 'v_edit.lam=1.0',
                                                             'v_edit.blocks=all', 'v_edit.steps=all']))
    check('pad_v_x1_all', config.load('klein', 'pad_v', ['pad_v.factor=1.0', 'pad_v.blocks=all', 'pad_v.steps=all']))
    check('dave_a1_block2', config.load('klein', 'dave', ['dave.alpha=1.0', 'dave.block=2', 'dave.steps=all']))
    check('dave_a1_block13', config.load('klein', 'dave', ['dave.alpha=1.0', 'dave.block=13', 'dave.steps=all']))
    ims = check('ours_smoke', config.load('klein', 'ours', [f'v_edit.basis={bpath}', 'v_edit.lam=1.5', 'v_edit.blocks=5-24',
                                                            'v_edit.steps=0-4', 'pad_v.steps=0-4']), expect_same=False)
    check('dave_a0.2_smoke', config.load('klein', 'dave'), expect_same=False)
    res = config.ROOT/'outputs'/'parity'
    res.mkdir(parents=True, exist_ok=True)
    for i, (r, o) in enumerate(zip(ref, ims)):
        r.save(res/f'klein_native_{SEEDS[i]}.png')
        o.save(res/f'klein_ours_smoke_{SEEDS[i]}.png')
    if x.speed:
        seeds = list(range(50000, 50016))
        t = time.monotonic()
        models.generate(pipe, m, emb, seeds)
        out['speed_original_batch16_s_per_image'] = (time.monotonic()-t)/16
        out['peak_gib'] = torch.cuda.max_memory_allocated(x.gpu)/2**30
        print('speed', out['speed_original_batch16_s_per_image'], 'peak', out['peak_gib'], flush=True)
    out['all_pass'] = all(v['pass'] for v in out.values() if isinstance(v, dict) and 'pass' in v)
    (res/'klein.json').write_text(json.dumps(out, indent=1))
    print('ALL_PASS' if out['all_pass'] else 'SOME_FAILED', flush=True)


if __name__ == '__main__':
    main()
