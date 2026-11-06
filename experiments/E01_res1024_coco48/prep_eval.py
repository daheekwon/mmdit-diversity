"""Evaluation folders of PROTOCOL_ko.md section 4.

  link512 <model> [--modes ...] [--wait]: outputs/res512/<model>/coco48/generation gets the legacy 512 COCO images of seeds
      68000-68007 as symlinks (modes whose pixels this package reproduces; see PROTOCOL_ko.md change log), meta.json with
      those seeds. --wait: first wait until the legacy run holds every image and meta.json of these modes.
  down1024 <model>: outputs/res1024/<model>/coco48/eval512 = the 1024 images resized to 512 (Lanczos) + meta.json.
      The judge, MUSIQ, CLIP and DINO read this copy; the sheets read the 1024 originals.
"""
import argparse
import json
from pathlib import Path
import time
from PIL import Image

PKG = Path(__file__).resolve().parents[2]
LEGACY = PKG.parent/'value_diversity'/'results'/'generalize'
SEEDS = range(68000, 68008)
LINKED = dict(sd35=['original', 'dave_a0.5'], flux=['b8_original', 'dave_a0.2', 'b8_q_d0.7+pad_v0.25'])


def link512(model, modes, wait=False):
    src, dst = LEGACY/model/'coco'/'generation', PKG/'outputs'/'res512'/model/'coco48'/'generation'
    need = [src/f'coco_{i:02d}'/m/f for i in range(48) for m in modes for f in ['meta.json']+[f'seed{s}.png' for s in SEEDS]]
    while wait and not all(p.exists() for p in need):
        time.sleep(60)
    for pid_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for mode in modes:
            s, d = pid_dir/mode, dst/pid_dir.name/mode
            d.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                link = d/f'seed{seed}.png'
                if not link.is_symlink():
                    link.symlink_to(s/f'seed{seed}.png')
            meta = json.loads((s/'meta.json').read_text())
            meta.update(seeds=list(SEEDS), linked_from=str(s))
            (d/'meta.json').write_text(json.dumps(meta, indent=1))
    print('linked', dst, modes)


def down1024(model):
    src = PKG/'outputs'/'res1024'/model/'coco48'/'generation'
    dst = src.parent/'eval512'
    n = 0
    for meta in sorted(src.glob('*/*/meta.json')):
        rel = meta.parent.relative_to(src)
        (dst/rel).mkdir(parents=True, exist_ok=True)
        for p in sorted(meta.parent.glob('seed*.png')):
            q = dst/rel/p.name
            if not q.exists():
                with Image.open(p) as im:
                    im.convert('RGB').resize((512, 512), Image.LANCZOS).save(q)
                n += 1
        (dst/rel/'meta.json').write_text(meta.read_text())
    print('resized', n, 'new images into', dst)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('action', choices=['link512', 'down1024'])
    a.add_argument('model', choices=['sd35', 'flux'])
    a.add_argument('--modes', nargs='*')
    a.add_argument('--wait', action='store_true')
    x = a.parse_args()
    if x.action == 'link512':
        link512(x.model, x.modes or LINKED[x.model], x.wait)
    else:
        down1024(x.model)


if __name__ == '__main__':
    main()
