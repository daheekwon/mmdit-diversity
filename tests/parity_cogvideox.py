"""GPU parity checks of the CogVideoX adapter against the native diffusers pipeline (no stored references exist).

  python tests/parity_cogvideox.py [--gpu auto] [--frames 9]

One prompt (the first VBench tune prompt), 2 seeds, short videos (--frames, default 9: 3 latent frames) so that the
checks are fast; the adapter code does not depend on the frame count. 'same' = every pixel of every frame equal.
  native pipeline (prompt string, one seed per call)    vs package encode + generate, no controller    expected: same
  v_edit lam 1 (dummy basis, every block and step: the adapter's own q / k / v and attention path)
                                                        vs package original                            expected: same
  dave alpha 1                                          vs package original                            expected: same
  pad_v factor 1 (video and text query rows attended in two calls)   vs original             information
  batch 2                                               vs batch 1                                     information
  smoke runs (the controller's call counts are verified): pad_v 1/8, dave 0.5, cads                    information
With --gpu auto the script waits until a GPU has enough free memory on two checks 30 s apart (shared machine).
Results: outputs/parity/cogvideox.json.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import config, data, models                                          # noqa: E402
from div.method import Controller                                             # noqa: E402
from parity import wait_gpu                                                   # noqa: E402

NEED = 12.                                                                    # GiB


def compare(a, b):
    d = np.abs(np.stack(a).astype(np.int16)-np.stack(b).astype(np.int16))
    return dict(same=bool((d == 0).all()), mean_abs=float(d.mean()), max_abs=int(d.max()))


def dummy_basis(path, dim, k=256):
    g = torch.Generator().manual_seed(0)
    Q = {b: torch.linalg.qr(torch.randn(dim, k, generator=g))[0] for b in range(models.N_BLOCKS['cogvideox'])}
    torch.save(dict(model='cogvideox', k=k, beta=1., blocks=sorted(Q), Q=Q, unit={b: 1. for b in Q}, check={},
                    source='random orthonormal, parity tests only'), path)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--gpu', default='auto')
    a.add_argument('--frames', type=int, default=9)
    x = a.parse_args()
    import parity
    parity.NEED['cogvideox'] = NEED
    gpu = wait_gpu('cogvideox') if x.gpu == 'auto' else int(x.gpu)
    out = config.ROOT/'outputs'/'parity'
    out.mkdir(parents=True, exist_ok=True)
    frames = [f'model.num_frames={x.frames}']
    base = config.load('cogvideox', 'original', frames)
    prompt = data.prompts(base.data, 'vbench_tune', samples=2)[0]['prompt']
    seeds = [5100000, 5100001]
    print(f'GPU {gpu}', flush=True)
    pipe = models.load(base.model, gpu)
    basis = out/'cogvideox_dummy_basis.pt'
    dummy_basis(basis, models.attentions(pipe, 'cogvideox')[0].to_v.out_features)
    emb = models.encode(pipe, base.model, prompt)

    def gen(preset, overrides=(), batch=1):
        cfg = config.load('cogvideox', preset, [*frames, *overrides])
        vids, info = [], {}
        for k in range(0, len(seeds), batch):
            ctrl = Controller(pipe, cfg, prompt, seeds[k:k+batch]) if preset != 'original' else None
            vids += models.generate(pipe, cfg.model, emb, seeds[k:k+batch], ctrl)
            info = ctrl.info() if ctrl else {}
        return vids, info

    def native():
        m = base.model
        vids = []
        for s in seeds:
            with torch.inference_mode():
                f = pipe(prompt=prompt, height=m.height, width=m.width, num_frames=m.num_frames, num_inference_steps=m.steps,
                         guidance_scale=m.guidance, use_dynamic_cfg=m.dynamic_cfg, max_sequence_length=m.max_sequence_length,
                         generator=[torch.Generator('cpu').manual_seed(s)], output_type='np').frames
            vids.append((f[0]*255).round().astype(np.uint8))
        return vids
    res = dict(prompt=prompt, seeds=seeds, frames=x.frames)
    t0 = time.monotonic()
    orig, _ = gen('original')
    res['native pipeline vs package original'] = compare(native(), orig)
    lam1, info = gen('v_edit', [f'v_edit.basis={basis}', 'v_edit.lam=1', 'v_edit.blocks=all', 'v_edit.steps=all'])
    res['v_edit lam1 (all blocks, all steps) vs original'] = dict(compare(lam1, orig), **info)
    res['dave alpha1 vs original'] = compare(gen('dave', ['dave.alpha=1'])[0], orig)
    res['pad_v factor1 vs original (info)'] = compare(gen('pad_v', ['pad_v.factor=1'])[0], orig)
    res['batch 2 vs batch 1 (info)'] = compare(gen('original', batch=2)[0], orig)
    for preset in ('pad_v', 'dave', 'cads'):
        vids, info = gen(preset)
        res[f'{preset} smoke vs original (info)'] = dict(compare(vids, orig), **info)
    res['seconds'] = time.monotonic()-t0
    (out/'cogvideox.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
