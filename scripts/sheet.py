"""Contact sheets of runs that share prompts and seeds.

  python scripts/sheet.py side outputs/sd35/coco_tune/{original,ours_legacy,ours} --prompts 10 --out sheets/sd35_coco
  python scripts/sheet.py blind outputs/sd35/coco/{original,ours,dave,cads} --prompts 10 --out blind/sd35_coco
  python scripts/sheet.py score blind/sd35_coco                     # after labels.csv is filled in

side   one block per prompt: a row per run (named), a column per seed; --per-sheet prompts per file <out>_<k>.jpg.
blind  one sheet per prompt: the images of all runs shuffled into one numbered grid, run names hidden.
       <out>/key.csv maps (sheet, cell) to run / prompt / seed; <out>/labels.csv has an empty label column for the
       labeller (empty = no failure, otherwise the failure type).
score  failures per run from key.csv + labels.csv.
Prompts: a fixed shuffle (random.Random(--seed)) of the prompts that every run has complete.
"""
import argparse
import csv
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div.metrics import run_rows                                              # noqa: E402

FONT = ImageFont.load_default(size=16)


def sample(runs, n, seed):
    rows = [{r['id']: r for r in run_rows(run) if all(f.exists() for f in r['files'])} for run in runs]
    common = sorted(set.intersection(*(set(r) for r in rows)))
    random.Random(seed).shuffle(common)
    return [[rr[i] for rr in rows] for i in common[:n]]


def thumb(path, t):
    return Image.open(path).convert('RGB').resize((t, t), Image.BICUBIC)


def header(text, width):
    lines = textwrap.wrap(text, width//9) or ['']
    im = Image.new('RGB', (width, 8+20*len(lines)), 'white')
    ImageDraw.Draw(im).multiline_text((6, 4), '\n'.join(lines), fill='black', font=FONT, spacing=4)
    return im


def stack(parts):
    out = Image.new('RGB', (max(p.width for p in parts), sum(p.height for p in parts)), 'white')
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    return out


def side(runs, prompts, out, t, per_sheet):
    names = [Path(r).name for r in runs]
    lab = 12+9*max(len(n) for n in names)
    for k in range(0, len(prompts), per_sheet):
        parts = []
        for group in prompts[k:k+per_sheet]:
            r0 = group[0]
            block = Image.new('RGB', (lab+t*len(r0['seeds']), t*len(runs)), 'white')
            d = ImageDraw.Draw(block)
            for i, (name, r) in enumerate(zip(names, group)):
                d.text((6, i*t+t//2-8), name, fill='black', font=FONT)
                for j, f in enumerate(r['files']):
                    block.paste(thumb(f, t), (lab+j*t, i*t))
            parts += [header(f"{r0['id']}: {r0['prompt']}", block.width), block]
        path = Path(f'{out}_{k//per_sheet+1}.jpg')
        path.parent.mkdir(parents=True, exist_ok=True)
        stack(parts).save(path, quality=88)
        print(path)


def blind(runs, prompts, out, t, seed, cols=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    key, rng = [], random.Random(seed+1)
    for s, group in enumerate(prompts, start=1):
        cells = [(Path(run).name, r['id'], seed_, f) for run, r in zip(runs, group) for seed_, f in zip(r['seeds'], r['files'])]
        rng.shuffle(cells)
        rows = (len(cells)+cols-1)//cols
        grid = Image.new('RGB', (cols*t, rows*t), 'white')
        d = ImageDraw.Draw(grid)
        for c, (run, pid, sd, f) in enumerate(cells, start=1):
            x, y = (c-1) % cols*t, (c-1)//cols*t
            grid.paste(thumb(f, t), (x, y))
            d.rectangle((x, y, x+34, y+20), fill='white')
            d.text((x+3, y+2), str(c), fill='black', font=FONT)
            key.append(dict(sheet=s, cell=c, run=run, id=pid, seed=sd))
        stack([header(f'sheet {s}: {group[0]["prompt"]}', grid.width), grid]).save(out/f'sheet_{s:02d}.jpg', quality=90)
    with open(out/'key.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['sheet', 'cell', 'run', 'id', 'seed'])
        w.writeheader()
        w.writerows(key)
    labels = out/'labels.csv'
    if not labels.exists():
        with open(labels, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['sheet', 'cell', 'label'])
            w.writerows([k['sheet'], k['cell'], ''] for k in key)
    print(f'{out}: {len(prompts)} sheets, {len(key)} images')


def score(out):
    out = Path(out)
    key = {(r['sheet'], r['cell']): r for r in csv.DictReader(open(out/'key.csv'))}
    labels = {(r['sheet'], r['cell']): r['label'].strip() for r in csv.DictReader(open(out/'labels.csv'))}
    if set(labels) != set(key):
        raise SystemExit('labels.csv does not cover exactly the cells of key.csv')
    n, fail, kinds = Counter(), Counter(), {}
    for c, k in key.items():
        n[k['run']] += 1
        if labels[c]:
            fail[k['run']] += 1
            kinds.setdefault(k['run'], Counter())[labels[c]] += 1
    print('| run | images | failures | rate | types |\n|---|---|---|---|---|')
    for run in sorted(n):
        print(f"| {run} | {n[run]} | {fail[run]} | {fail[run]/n[run]:.1%} | "
              f"{', '.join(f'{k} {v}' for k, v in kinds.get(run, Counter()).most_common())} |")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('mode', choices=['side', 'blind', 'score'])
    a.add_argument('runs', nargs='*')
    a.add_argument('--prompts', type=int, default=10)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--size', type=int, default=160)
    a.add_argument('--per-sheet', type=int, default=5)
    a.add_argument('--out', required=False)
    x = a.parse_args()
    if x.mode == 'score':
        return score(x.runs[0] if x.runs else x.out)
    prompts = sample(x.runs, x.prompts, x.seed)
    if len(prompts) < x.prompts:
        print(f'only {len(prompts)} prompts complete in every run', flush=True)
    if x.mode == 'side':
        side(x.runs, prompts, x.out, x.size, x.per_sheet)
    else:
        blind(x.runs, prompts, x.out, x.size, x.seed)


if __name__ == '__main__':
    main()
