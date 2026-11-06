"""Evaluation folders of PROTOCOL_ko.md section 5.

  link <model>: Original and DAVE of E01 (512 generation, 1024 generation and eval512) appear in the E04 folders:
      real folders, seed PNGs as symlinks, meta.json copied.
  down <model>: the final method's 1024 images resized to 512 (Lanczos) into eval512 + meta.json.
"""
import argparse
from pathlib import Path
import sys
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAVE, FINAL, ORIG, SEEDS, e01_root, eval_root, gen_root    # noqa: E402


def link_tree(src, dst, modes):
    n = 0
    for pid in sorted(p.name for p in src.iterdir() if p.is_dir()):
        for mode in modes:
            s, d = src/pid/mode, dst/pid/mode
            d.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                link = d/f'seed{seed}.png'
                if not (s/link.name).exists():
                    raise FileNotFoundError(s/link.name)
                if not link.is_symlink():
                    link.symlink_to((s/link.name).resolve())
                    n += 1
            (d/'meta.json').write_text((s/'meta.json').read_text())
    print('linked', n, 'images', src, '->', dst, modes, flush=True)


def link(model):
    modes = [ORIG[model], DAVE[model]]
    link_tree(e01_root(model, 512), gen_root(model, 512), modes)
    link_tree(e01_root(model, 1024), gen_root(model, 1024), modes)
    link_tree(e01_root(model, 1024, 'eval512'), eval_root(model, 1024), modes)


def down(model):
    src, dst, n = gen_root(model, 1024), eval_root(model, 1024), 0
    for meta in sorted(src.glob(f'*/{FINAL[model]}/meta.json')):
        rel = meta.parent.relative_to(src)
        (dst/rel).mkdir(parents=True, exist_ok=True)
        for p in sorted(meta.parent.glob('seed*.png')):
            q = dst/rel/p.name
            if not q.exists():
                with Image.open(p) as im:
                    im.convert('RGB').resize((512, 512), Image.LANCZOS).save(q)
                n += 1
        (dst/rel/'meta.json').write_text(meta.read_text())
    print('resized', n, 'new images into', dst, flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('action', choices=['link', 'down'])
    a.add_argument('model', choices=['sd35', 'flux'])
    x = a.parse_args()
    (link if x.action == 'link' else down)(x.model)


if __name__ == '__main__':
    main()
