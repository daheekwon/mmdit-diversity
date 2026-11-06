"""Pixel comparison of two generation folders <root>/<pid>/<mode>/seedXXXXX.png over the images both hold.

  python experiments/E01_res1024_coco48/pixel_match.py <new root> <reference root>
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def main(new, ref):
    new, ref, bad = Path(new), Path(ref), 0
    for folder in sorted(p.parent for p in new.glob('*/*/meta.json')):
        rel = folder.relative_to(new)
        pairs = [(p, ref/rel/p.name) for p in sorted(folder.glob('seed*.png')) if (ref/rel/p.name).exists()]
        diff = [np.abs(np.asarray(Image.open(a), np.int16)-np.asarray(Image.open(b), np.int16)) for a, b in pairs]
        same = sum(int(d.max() == 0) for d in diff)
        worst = max((float(d.mean()) for d in diff), default=float('nan'))
        bad += same < len(pairs) or not pairs
        print(f'{rel}: {same}/{len(pairs)} identical, worst mean |diff| {worst:.4f}')
    print('ALL IDENTICAL' if not bad else f'{bad} folders differ or have no pairs')


if __name__ == '__main__':
    main(*sys.argv[1:3])
