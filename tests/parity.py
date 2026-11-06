"""GPU parity checks against the earlier value_diversity/ images (prompt dev_animal_00, 8 seeds).

  python tests/parity.py --model sd35 [--gpu auto]

Each check generates with this package and compares pixels with a reference ('same' = every pixel equal):
  sd35  original                 vs vpad_grid original                                   expected: same
        dave a0.5 (block 6)      vs vpad_grid dave_a0.5                                  expected: same
        v_edit lam 1             vs this package's original                               expected: same
        ours, old point          vs the legacy runner's single-pass mode (--fast), run now expected: same
          (lam 1.4 on 6-23, V 0-3, pad 1/8 0-2; and vs the stored hook-implementation images, for information)
        ours, config default     vs the legacy single-pass qa_l1.14@0-23~0-11+padv0.25~0-11, run now  expected: same
          (lam 1.14 on 0-23, V and pad 1/4 0-11; the generalize/allblock images are hook images: information)
        cads, ours_head          smoke runs (the legacy SD3.5 CADS drew its noise per batch, so no reference)
  flux  original (batch 8)       vs vpad_grid b8_original                                 expected: same
        ours, old point          vs vpad_grid b8_q_d0.7+pad_v0.25 (d0.7 = lam 2.01 on 19-37, V 0-5, pad 1/4 0-2;
                                 hook = single pass in FLUX)                              expected: same
        ours d0.5, V and pad 0-5 vs generalize flux/coco coco_00 b8_q_d0.5+pad_v0.25~0-5   expected: same
        ours, config default     vs the same images (lam 1.72; the legacy d0.5 is lam 1.7196): information
        dave a0.2 / cads (batch 1) vs vpad_grid dave_a0.2 / cads_n0.25 (made one seed at a time)  expected: same
        ours_head                smoke run
Every run is at 512 (model.resolution=512, the resolution of the reference images; the config default is 1024).
With --gpu auto the script waits until a GPU has enough free memory on two checks 30 s apart (shared machine).
Results: outputs/parity/<model>.json.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import config, models                                                # noqa: E402
from div.method import Controller                                             # noqa: E402

LEGACY = config.ROOT.parent/'value_diversity'
GRID = LEGACY/'results'/'vpad_grid'
GEN = LEGACY/'results'/'generalize'
RES = 'model.resolution=512'
SD35_NEW = 'qa_l1.14@0-23~0-11+padv0.25~0-11'                                # legacy name of the sd35 config default
OLD = dict(sd35=['v_edit.basis=bases/sd35_legacy_dev8_k256.pt', 'v_edit.lam=1.4', 'v_edit.blocks=6-23',
                 'v_edit.steps=0-3', 'pad_v.steps=0-2'],                      # operating points of 11-03
           flux=['v_edit.lam=null', 'v_edit.strength=0.7', 'v_edit.blocks=19-37', 'v_edit.steps=0-5', 'pad_v.steps=0-2'])
PY = sys.executable
NEED = dict(sd35=20., flux=36.)                                               # GiB


def wait_gpu(model):
    def free():
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True).stdout.strip().splitlines()
        return {int(i): (float(t)-float(u))/1024 for i, u, t in (l.split(',') for l in out)}
    while True:
        a = free()
        time.sleep(30)
        b = free()
        ok = [g for g in a if min(a[g], b[g]) >= NEED[model]]
        if ok:
            return max(ok, key=lambda g: b[g])
        print(f'waiting for {NEED[model]:.0f} GiB free: {b}', flush=True)
        time.sleep(90)


def compare(a, b):
    a = np.stack([np.asarray(Image.open(p) if isinstance(p, (str, Path)) else p, dtype=np.int16) for p in a])
    b = np.stack([np.asarray(Image.open(p) if isinstance(p, (str, Path)) else p, dtype=np.int16) for p in b])
    d = np.abs(a-b)
    return dict(same=bool((d == 0).all()), mean_abs=float(d.mean()), max_abs=int(d.max()))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=['sd35', 'flux'])
    a.add_argument('--gpu', default='auto')
    x = a.parse_args()
    gpu = wait_gpu(x.model) if x.gpu == 'auto' else int(x.gpu)
    out = config.ROOT/'outputs'/'parity'
    out.mkdir(parents=True, exist_ok=True)
    pid = 'dev_animal_00'                         # SD3.5 'A dog', FLUX 'A red panda'
    prompt = json.loads((GRID/x.model/'generation'/pid/('original' if x.model == 'sd35' else 'b8_original')/'meta.json')
                        .read_text())['prompt']
    seeds = list(range(20000, 20008)) if x.model == 'sd35' else list(range(50000, 50008))
    ref = lambda mode: [GRID/x.model/'generation'/pid/mode/f'seed{s}.png' for s in seeds]
    legacy_fast = None
    if x.model == 'sd35':                         # legacy single-pass run, before this package loads its pipeline
        legacy_fast = out/'legacy_fast'
        modes = [m for m in ('q_l1.4+padv0.125', SD35_NEW) if not (legacy_fast/pid/m/'meta.json').exists()]
        if modes:
            subprocess.run([PY, '-m', 'value_diversity.combo_sd35', 'run', '--gpu', str(gpu), '--prompts', pid,
                            '--modes', *modes, '--fast', '--out', str(legacy_fast)], cwd=LEGACY.parent, check=True)
    print(f'GPU {gpu}', flush=True)
    base = config.load(x.model, 'original', ['model.batch_size=8', RES])
    pipe = models.load(base.model, gpu)
    emb = models.encode(pipe, base.model, prompt)

    def gen(preset, overrides=(), batch=8, other=None):
        """other: (pid, prompt, seeds) instead of the dev_animal_00 ones."""
        pid_, prompt_, seeds_ = other or (pid, prompt, seeds)
        e = models.encode(pipe, base.model, prompt_) if other else emb
        cfg = config.load(x.model, preset, [f'model.batch_size={batch}', RES, *overrides])
        ims, info = [], {}
        for k in range(0, len(seeds_), batch):
            ctrl = Controller(pipe, cfg, prompt_, seeds_[k:k+batch]) if preset != 'original' else None
            ims += models.generate(pipe, cfg.model, e, seeds_[k:k+batch], ctrl)
            info = ctrl.info() if ctrl else {}
        sub = out/x.model/(pid_ if other else '')/config.tag(cfg)
        sub.mkdir(parents=True, exist_ok=True)
        for s, im in zip(seeds_, ims):
            im.save(sub/f'{s}.png')
        return ims, info

    def hook_seeds(folder):
        return json.loads((folder/'meta.json').read_text())['seeds'][:8]

    def gen_prompt(folder, overrides=()):
        """'ours' on the prompt and first 8 seeds of a legacy generation folder <pid>/<mode>."""
        return gen('ours', overrides, other=(folder.parent.name, json.loads((folder/'meta.json').read_text())['prompt'],
                                             hook_seeds(folder)))
    res = {}
    t0 = time.monotonic()
    orig, _ = gen('original')
    if x.model == 'sd35':
        res['original vs legacy'] = compare(orig, ref('original'))
        res['dave a0.5 vs legacy'] = compare(gen('dave')[0], ref('dave_a0.5'))
        res['v_edit lam1 vs original'] = compare(gen('v_edit', ['v_edit.lam=1'])[0], orig)
        ours, info = gen('ours', OLD['sd35'])
        res['ours vs legacy single-pass'] = dict(compare(ours, [legacy_fast/pid/'q_l1.4+padv0.125'/f'seed{s}.png' for s in seeds]), **info)
        new, info = gen('ours')
        res['default vs legacy single-pass qa_l1.14'] = dict(compare(new, [legacy_fast/pid/SD35_NEW/f'seed{s}.png' for s in seeds]), **info)
        hook = GEN/'sd35'/'tune'/'generation'/'ho3_relation_00'/SD35_NEW
        res['default vs generalize hook images (info)'] = compare(gen_prompt(hook)[0], [hook/f'seed{s}.png' for s in hook_seeds(hook)])
        res['ours vs legacy hook (info)'] = compare(ours, ref('q_l1.4+padv0.125'))
        res['legacy single-pass vs legacy hook (info)'] = compare([legacy_fast/pid/'q_l1.4+padv0.125'/f'seed{s}.png' for s in seeds],
                                                                  ref('q_l1.4+padv0.125'))
        res['cads smoke vs original (info)'] = compare(gen('cads')[0], orig)
    else:
        res['original vs legacy b8'] = compare(orig, ref('b8_original'))
        ours, info = gen('ours', OLD['flux'])
        res['ours vs legacy b8_q_d0.7+pad_v0.25'] = dict(compare(ours, ref('b8_q_d0.7+pad_v0.25')), **info)
        final = GEN/'flux'/'coco'/'generation'/'coco_00'/'b8_q_d0.5+pad_v0.25~0-5'
        d05, info = gen_prompt(final, ['v_edit.lam=null', 'v_edit.strength=0.5'])
        res['d0.5 V and pad 0-5 vs generalize coco_00'] = dict(compare(d05, [final/f'seed{s}.png' for s in hook_seeds(final)]), **info)
        new, info = gen_prompt(final)
        res['default lam 1.72 vs generalize coco_00 (info)'] = dict(compare(new, [final/f'seed{s}.png' for s in hook_seeds(final)]), **info)
        res['dave a0.2 batch1 vs legacy'] = compare(gen('dave', batch=1)[0], ref('dave_a0.2'))
        res['cads batch1 vs legacy'] = compare(gen('cads', batch=1)[0], ref('cads_n0.25'))
    res['ours_head smoke vs ours (info)'] = compare(gen('ours_head', OLD[x.model])[0], ours)
    res['seconds'] = time.monotonic()-t0
    (out/f'{x.model}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
