"""Generate E06 conditions (PROTOCOL_ko.md) in the layout value_diversity's evaluation reads:
<root>/<pid>/<mode>/seedXXXXX.png + meta.json, at 512.

  python experiments/E06_boundary/run_e06.py --model sd35 --split e06 --shard 0/8 --gpu 0
  python experiments/E06_boundary/run_e06.py --model flux --split ho4 --modes v_l1.72 --pids ho4_animal_00 --root <scratch>

Existing images are skipped, so a retried job resumes. --shard i/n splits the prompts across processes.
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

BATCH = 8


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=['sd35', 'flux'])
    a.add_argument('--split', required=True, choices=['e06', 'ho4'])
    a.add_argument('--modes', nargs='*', help='default: every condition of the split')
    a.add_argument('--pids', nargs='*')
    a.add_argument('--seeds', nargs='*', type=int)
    a.add_argument('--root')
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--shard', default='0/1')
    x = a.parse_args()
    table = {k: (p, o) for k, (p, o, twin) in C.all_modes(x.model).items() if not twin}   # stage 1 + random direction
    names = x.modes or (C.E06_MODES if x.split == 'e06' else C.HO4_MODES)[x.model]
    if 'RANDOM' in names:                      # queue placeholder: the random condition, named by its measured lam_r
        rm = C.random_mode(x.model)
        if rm is None:
            raise SystemExit('no write measure yet (run_probe.py write)')
        names = [rm[0] if n == 'RANDOM' else n for n in names]
    seeds = x.seeds or C.SEEDS[x.split]
    root = Path(x.root) if x.root else C.gen_root(x.model, x.split)
    rows = [r for r in C.prompts(x.split) if not x.pids or r['id'] in x.pids]
    i, n = map(int, x.shard.split('/'))
    rows = rows[i::n]
    base = config.load(x.model, 'original', [f'model.resolution={C.RES}'])
    pipe = models.load(base.model, x.gpu)
    print(f'{root}: {len(rows)} prompts, modes {names}, {C.RES}px', flush=True)
    for r in rows:
        emb = models.encode(pipe, base.model, r['prompt'])
        for mode in names:
            preset, over = table[mode]
            cfg = config.load(x.model, preset, [f'model.resolution={C.RES}', *over])
            folder = root/r['id']/mode
            todo = [s for s in seeds if not (folder/f'seed{s}.png').exists()]
            if not todo:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            t0, info = time.monotonic(), {}
            for k in range(0, len(todo), BATCH):
                part = todo[k:k+BATCH]
                ctrl = Controller(pipe, cfg, r['prompt'], part) if preset != 'original' else None
                for s, im in zip(part, models.generate(pipe, cfg.model, emb, part, ctrl)):
                    im.save(folder/f'seed{s}.png')
                info = ctrl.info() if ctrl is not None else {}
            seen = sorted(int(p.stem[4:]) for p in folder.glob('seed*.png'))
            meta = dict(pid=r['id'], prompt=r['prompt'], group=r['group'], mode=mode, seeds=seen, preset=preset,
                        overrides=over, resolution=C.RES, batch=BATCH,
                        method={c: dict(cfg.method[c]) for c in config.COMPONENTS if cfg.method[c].enabled},
                        info=info, seconds=time.monotonic()-t0)
            (folder/'meta.json').write_text(json.dumps(meta, indent=1, default=str))
            print(f"SAVED {r['id']} {mode} {len(todo)} images {meta['seconds']:.1f}s", flush=True)
    print('GEN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
