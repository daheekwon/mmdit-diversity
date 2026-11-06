"""E06 image sheets (PROTOCOL §7): one prompt per figure, rows = conditions (labelled with the table names), columns =
seeds; red frame = judge broken (defect or style >= .5), orange = mismatch (match < .5). Also a sheet of judge-broken images
per condition. Output: outputs/e06_boundary/<model>/<split>/sheets/.

  python experiments/E06_boundary/sheets.py sd35 e06 --pids tune_00 comp_03 rare_kinkajou
  python experiments/E06_boundary/sheets.py sd35 e06 --broken v_l1.14 v_l1.14+padv0.25 dave_a0.5
"""
import argparse
import csv
from pathlib import Path
import random
import sys
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402

ROWS = dict(sd35=['original', 'v_l1.14', 'v_l1.2', 'v_l1.3', 'padv0.25', 'padv0', 'v_l1.14+padv0.25', 'dave_a0.5', 'cads0.25'],
            flux=['original', 'v_l1.72', 'v_l2.01', 'v_l2.58', 'padv0.25', 'padv0', 'v_l1.72+padv0.25', 'dave_a0.2', 'cads0.25'])
W, LAB = 128, 190


def judge(root):
    J = {}
    for f in sorted(root.glob('judge_e06_*of2.csv')):
        for r in csv.DictReader(f.open()):
            J[(r['pid'], r['mode'], r['seed'])] = (float(r['p_defect']) >= .5 or float(r['p_style']) >= .5,
                                                   float(r['p_match']) < .5)
    return J


def font(size=14):
    for p in ('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def tile(img_path, flags):
    im = Image.open(img_path).convert('RGB').resize((W, W))
    d = ImageDraw.Draw(im)
    if flags and flags[0]:
        d.rectangle([0, 0, W-1, W-1], outline=(220, 0, 0), width=4)
    elif flags and flags[1]:
        d.rectangle([0, 0, W-1, W-1], outline=(255, 150, 0), width=4)
    return im


def prompt_sheet(model, split, pid, rows, J, out):
    gen = C.gen_root(model, split)
    seeds = C.SEEDS[split][:8]
    prompt = next(r['prompt'] for r in C.prompts(split) if r['id'] == pid)
    rows = [m for m in rows if (gen/pid/m).exists()]
    sheet = Image.new('RGB', (LAB+W*len(seeds), 30+W*len(rows)), 'white')
    d, f = ImageDraw.Draw(sheet), font()
    d.text((6, 6), f'{model} {split} {pid}: {prompt}'[:150], fill='black', font=f)
    for i, m in enumerate(rows):
        d.text((6, 30+i*W+W//2-8), m, fill='black', font=f)
        for j, s in enumerate(seeds):
            p = gen/pid/m/f'seed{s}.png'
            if p.exists():
                sheet.paste(tile(p, J.get((pid, m, f'seed{s}'))), (LAB+j*W, 30+i*W))
    sheet.save(out/f'{pid}.jpg', quality=88)


def broken_sheet(model, split, modes, J, out, n=24):
    gen = C.gen_root(model, split)
    rng = random.Random(0)
    sheet = Image.new('RGB', (LAB+W*n//2, 30+W*2*len(modes)), 'white')
    d, f = ImageDraw.Draw(sheet), font()
    d.text((6, 6), f'{model} {split}: judge-broken images (random {n} per condition, seed 0)', fill='black', font=f)
    for i, m in enumerate(modes):
        ks = sorted(k for k, v in J.items() if k[1] == m and v[0])
        pick = rng.sample(ks, min(n, len(ks)))
        d.text((6, 30+i*2*W+W-8), f'{m}\n({len(ks)} broken)', fill='black', font=f)
        for j, (pid, _, s) in enumerate(pick):
            sheet.paste(tile(gen/pid/m/f'{s}.png', None), (LAB+(j % (n//2))*W, 30+i*2*W+(j//(n//2))*W))
    sheet.save(out/f'broken_{"_".join(modes)[:80]}.jpg', quality=88)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('model')
    a.add_argument('split')
    a.add_argument('--pids', nargs='*', default=[])
    a.add_argument('--rows', nargs='*')
    a.add_argument('--broken', nargs='*', default=[])
    a.add_argument('--sub', default='', help='write into sheets/<sub>/ (keeps earlier sheets of the same prompt)')
    x = a.parse_args()
    root = C.gen_root(x.model, x.split).parent
    out = root/'sheets'/x.sub if x.sub else root/'sheets'
    out.mkdir(parents=True, exist_ok=True)
    J = judge(root)
    for pid in x.pids:
        prompt_sheet(x.model, x.split, pid, x.rows or ROWS[x.model], J, out)
    if x.broken:
        broken_sheet(x.model, x.split, x.broken, J, out)
    print(out)


if __name__ == '__main__':
    main()
