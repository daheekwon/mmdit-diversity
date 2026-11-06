"""Video contact sheets: one prompt per figure, a row per run (named as in the protocol table), a column per seed,
each cell the first, middle and last frame side by side.

  python scripts/video_sheet.py outputs/cogvideox/vbench_tune/{original,pad_v,v_edit,ours} \
      --names 원본 "pad V만" V편집만 "우리 방법" --out outputs/cogvideox/pilot/sheets
  ... --seeds 0-3 --suffix _s0-3          # a half sheet (seed positions 0..3)

Writes <out>/<prompt id><suffix>.jpg for every prompt that all runs have complete.
"""
import argparse
from pathlib import Path
import sys
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import config                                                        # noqa: E402
from div.video_metrics import read_video, run_rows                            # noqa: E402

FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
FRAMES = (0, 24, 48)
LABEL_W, GAP = 130, 6


def sheet(rows, names, seeds, width):
    h = round(width*480/720)
    cell_w = len(FRAMES)*width
    top = 36+22*len(textwrap.wrap(rows[0]['prompt'], 180))
    W, H = LABEL_W+len(seeds)*(cell_w+GAP), top+len(rows)*(h+GAP)
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    d.text((6, 4), f"{rows[0]['id']}  {rows[0]['short']}", fill='black', font=FONT)
    for k, line in enumerate(textwrap.wrap(rows[0]['prompt'], 180)):
        d.text((6, 28+22*k), line, fill='dimgray', font=ImageFont.truetype(FONT.path, 14))
    for i, (r, name) in enumerate(zip(rows, names)):
        y = top+i*(h+GAP)
        d.text((6, y+h//2-10), name, fill='black', font=FONT)
        for j, s in enumerate(seeds):
            v = read_video(r['files'][s])
            x = LABEL_W+j*(cell_w+GAP)
            for k, f in enumerate(FRAMES):
                im.paste(Image.fromarray(v[f]).resize((width, h), Image.BICUBIC), (x+k*width, y))
    return im


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('runs', nargs='+')
    a.add_argument('--names', nargs='+', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--seeds', default='all', help='seed positions, e.g. 0-3')
    a.add_argument('--width', type=int, default=160, help='frame width in the sheet')
    a.add_argument('--suffix', default='')
    x = a.parse_args()
    if len(x.names) != len(x.runs):
        a.error('one name per run')
    runs = [{r['id']: r for r in run_rows(run) if all(f.exists() for f in r['files'])} for run in x.runs]
    out = Path(x.out)
    out.mkdir(parents=True, exist_ok=True)
    for pid in sorted(set.intersection(*(set(r) for r in runs))):
        rows = [r[pid] for r in runs]
        seeds = config.parse_range(x.seeds, len(rows[0]['seeds']))
        sheet(rows, x.names, seeds, x.width).save(out/f'{pid}{x.suffix}.jpg', quality=88)
        print(pid, flush=True)


if __name__ == '__main__':
    main()
