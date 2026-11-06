"""Generate one run: a model x a dataset x a method setting.

  python scripts/generate.py --model sd35 --dataset coco --preset ours v_edit.strength=0.35 pad_v.factor=0.25
  python scripts/generate.py --model flux --dataset imagenet --preset dave --n-prompts 50      # tune subset

Output: outputs/<model>/<dataset>/<tag>/ with images/<prompt id>/<seed>.png (video models: videos/<prompt id>/<seed>.mp4),
prompts.jsonl and run.json.
<tag> = preset plus overrides, or --name. Existing images are skipped, so a run resumes where it stopped and a tune
subset fills the same folder that the full run completes later. --shard i/n splits the prompts across processes.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import config, data, models                                          # noqa: E402
from div.method import Controller                                             # noqa: E402


def prepare(run, cfg, dataset, rows):
    """run.json (settings; a tag must always mean the same settings) and prompts.jsonl (all prompts asked so far)."""
    run.mkdir(parents=True, exist_ok=True)
    spec = dict(model=OmegaConf.to_container(cfg.model), method=OmegaConf.to_container(cfg.method))
    meta = run/'run.json'
    if meta.exists():
        old = json.loads(meta.read_text())
        if dict(model=old['config']['model'], method=old['config']['method']) != spec:
            raise SystemExit(f'{run} holds a run with different settings; use another --name')
    else:
        meta.write_text(json.dumps(dict(dataset=dataset, preset=cfg.preset, overrides=list(cfg.overrides),
                                        config=OmegaConf.to_container(cfg)), indent=1))
    path = run/'prompts.jsonl'
    have = {r['id']: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}
    have.update({r['id']: r for r in rows})
    path.write_text(''.join(json.dumps(r)+'\n' for r in sorted(have.values(), key=lambda r: r['index'])))


def save_video(path, frames, fps):
    """H.264 at CRF 12 (near-lossless), yuv420p. Written under a temporary name, so an interrupted write is redone."""
    import imageio.v2 as imageio
    tmp = path.with_name(path.stem+'.part.mp4')
    imageio.mimwrite(tmp, list(frames), fps=fps, codec='libx264', quality=None, pixelformat='yuv420p',
                     output_params=['-crf', '12'], macro_block_size=16)
    tmp.rename(path)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--model', required=True, choices=['sd35', 'flux', 'cogvideox', 'klein'])
    a.add_argument('--dataset', required=True, choices=data.DATASETS)
    a.add_argument('--preset', default='original')
    a.add_argument('overrides', nargs='*', help='e.g. v_edit.strength=0.35 pad_v.blocks=0-18 model.batch_size=8')
    a.add_argument('--n-prompts', type=int, help='tune subset: first N prompts of a fixed shuffle')
    a.add_argument('--samples', type=int, help='images per prompt (default data.samples_per_prompt)')
    a.add_argument('--name', help='run folder name (default: preset + overrides)')
    a.add_argument('--out', default=str(config.ROOT/'outputs'))
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--shard', default='0/1', help='i/n: this process takes prompts i, i+n, ...')
    x = a.parse_args()
    pause = config.ROOT/'outputs'/f'.pause_gpu{x.gpu}'      # scheduling: another session's job needs this GPU now;
    while pause.exists():                                    # wait before touching the GPU (lanes keeps the lane)
        time.sleep(30)
    cfg = config.load(x.model, x.preset, x.overrides)
    run = Path(x.out)/x.model/x.dataset/(x.name or config.tag(cfg))
    rows = data.prompts(cfg.data, x.dataset, x.n_prompts, x.samples)
    prepare(run, cfg, x.dataset, rows)
    edits = any(cfg.method[c].enabled for c in config.COMPONENTS)
    video = cfg.model.name in models.VIDEO
    kind, ext = ('videos', '.mp4') if video else ('images', '.png')
    pipe = models.load(cfg.model, x.gpu)
    i, n = map(int, x.shard.split('/'))
    print(f'{run}: {len(rows[i::n])} prompts', flush=True)
    for r in rows[i::n]:
        folder = run/kind/r['id']
        todo = [s for s in r['seeds'] if not (folder/f'{s}{ext}').exists()]
        if not todo:
            continue
        t0, emb = time.monotonic(), models.encode(pipe, cfg.model, r['prompt'])
        for k in range(0, len(todo), cfg.model.batch_size):
            seeds = todo[k:k+cfg.model.batch_size]
            ctrl = Controller(pipe, cfg, r['prompt'], seeds) if edits else None
            images = models.generate(pipe, cfg.model, emb, seeds, ctrl)
            folder.mkdir(parents=True, exist_ok=True)
            for s, im in zip(seeds, images):
                if video:
                    save_video(folder/f'{s}.mp4', im, cfg.model.fps)
                else:
                    im.save(folder/f'{s}.png')
        extra = ctrl.info() if ctrl is not None else {}
        print(f"{r['id']} {len(todo)} {kind} {time.monotonic()-t0:.1f}s {extra}", flush=True)


if __name__ == '__main__':
    main()
