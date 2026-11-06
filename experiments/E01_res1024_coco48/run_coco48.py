"""Generate the COCO 48 check (PROTOCOL_ko.md) with this package, in the layout value_diversity's evaluation reads:
<root>/<pid>/<mode>/seedXXXXX.png + meta.json. Mode names are the legacy ones, so 512 and 1024 runs pair by name.

  python experiments/E01_res1024_coco48/run_coco48.py --model sd35 --res 1024 --gpu 0
  python experiments/E01_res1024_coco48/run_coco48.py --model flux --res 512 --pids coco_00 --root <scratch>   # parity

Existing images are skipped. --shard i/n splits the prompts across processes.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG))
from div import config, models                                                # noqa: E402
from div.method import Controller                                             # noqa: E402

SPLIT = PKG.parent/'value_diversity'/'coco48_split.py'
# legacy mode name -> (preset, batch of the 512 run)
MODES = dict(
    sd35={'original': ('original', 8), 'dave_a0.5': ('dave', 8), 'q_l1.4+padv0.125': ('ours', 8)},
    flux={'b8_original': ('original', 8), 'dave_a0.2': ('dave', 1), 'b8_q_d0.7+pad_v0.25': ('ours', 8)})


def coco48():
    spec = importlib.util.spec_from_file_location('coco48_split', SPLIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=list(MODES))
    a.add_argument('--res', type=int, required=True)
    a.add_argument('--modes', nargs='*', help='default: the three conditions of the protocol')
    a.add_argument('--pids', nargs='*', help='default: coco_00..coco_47')
    a.add_argument('--seeds', nargs='*', type=int, default=list(range(68000, 68008)))
    a.add_argument('--batch', type=int, help='images per pipeline call (default: the batch of the 512 run)')
    a.add_argument('--root', help='default: outputs/res<res>/<model>/coco48/generation')
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--shard', default='0/1')
    x = a.parse_args()
    modes = x.modes or list(MODES[x.model])
    root = Path(x.root or PKG/'outputs'/f'res{x.res}'/x.model/'coco48'/'generation')
    rows = [r for r in coco48() if not x.pids or r['id'] in x.pids]
    i, n = map(int, x.shard.split('/'))
    rows = rows[i::n]
    base = config.load(x.model)
    base.model.resolution = x.res
    pipe = models.load(base.model, x.gpu)
    print(f'{root}: {len(rows)} prompts, modes {modes}, {x.res}px', flush=True)
    for r in rows:
        emb = models.encode(pipe, base.model, r['prompt'])
        for mode in modes:
            preset, batch = MODES[x.model][mode]
            batch = x.batch or batch
            cfg = config.load(x.model, preset, [f'model.resolution={x.res}'])
            folder = root/r['id']/mode
            todo = [s for s in x.seeds if not (folder/f'seed{s}.png').exists()]
            if not todo:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            t0, info = time.monotonic(), {}
            for k in range(0, len(todo), batch):
                seeds = todo[k:k+batch]
                ctrl = Controller(pipe, cfg, r['prompt'], seeds) if preset != 'original' else None
                for s, im in zip(seeds, models.generate(pipe, cfg.model, emb, seeds, ctrl)):
                    im.save(folder/f'seed{s}.png')
                info = ctrl.info() if ctrl is not None else {}
            seen = sorted(int(p.stem[4:]) for p in folder.glob('seed*.png'))
            meta = dict(pid=r['id'], prompt=r['prompt'], mode=mode, seeds=seen, preset=preset, resolution=x.res,
                        batch=batch, method={c: dict(cfg.method[c]) for c in config.COMPONENTS if cfg.method[c].enabled},
                        info=info, seconds=time.monotonic()-t0)
            (folder/'meta.json').write_text(json.dumps(meta, indent=1, default=str))
            print(f"SAVED {r['id']} {mode} {len(todo)} images {meta['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
