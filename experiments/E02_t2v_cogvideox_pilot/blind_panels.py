"""Blind panels for verdict 1 of PROTOCOL_ko.md section 7c, and the unblinding of the counts.

  python experiments/E02_t2v_cogvideox_pilot/blind_panels.py make
  python experiments/E02_t2v_cogvideox_pilot/blind_panels.py score        # after blind/counts.json is written
  python experiments/E02_t2v_cogvideox_pilot/blind_panels.py score --counts counts_clear.json   # another counts file

make: for each of the 4 prompts x 8 seeds, one panel with one row per condition (consecutive frames 20-27), rows in a
random order (fixed generator) and labelled A-E only. Panels: outputs/cogvideox/pilot/blind/<prompt>_<seed position>.jpg;
the order: blind/key.json (not to be read before the counts are written).
score: blind/counts.json = {"<prompt>_<seed position>": ["B", "D"], ...} (failing rows; every panel listed, [] if none)
-> failing videos per condition and prompt, written to blind/score_<counts file stem>.md.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from div import config                                                        # noqa: E402
from div.video_metrics import read_video, run_rows                            # noqa: E402

RUNS = config.ROOT/'outputs'/'cogvideox'/'vbench_tune'
OUT = config.ROOT/'outputs'/'cogvideox'/'pilot'/'blind'
CONDITIONS = {'original': '원본', 'ours': '우리 방법', 'ours_time_mean': '우리 방법(시간 평균)',
              'v_edit': 'V편집만', 'v_edit_time_mean': 'V편집만(시간 평균)'}
PROMPTS = ('vd730', 'vd214', 'vd354', 'vd598')
FRAMES = range(20, 28)
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
W, H, LABEL, GAP = 180, 120, 40, 6


def make():
    rows = {t: {r['id']: r for r in run_rows(RUNS/t)} for t in CONDITIONS}
    rng = np.random.default_rng(20261105)
    OUT.mkdir(parents=True, exist_ok=True)
    key = {}
    for pid in PROMPTS:
        for k in range(len(rows['original'][pid]['seeds'])):
            order = [list(CONDITIONS)[i] for i in rng.permutation(len(CONDITIONS))]
            name = f'{pid}_{k}'
            key[name] = dict(zip('ABCDE', order))
            im = Image.new('RGB', (LABEL+len(FRAMES)*W, len(order)*(H+GAP)), 'white')
            d = ImageDraw.Draw(im)
            for i, t in enumerate(order):
                v = read_video(rows[t][pid]['files'][k])
                y = i*(H+GAP)
                d.text((8, y+H//2-14), 'ABCDE'[i], fill='black', font=FONT)
                for j, f in enumerate(FRAMES):
                    im.paste(Image.fromarray(v[f]).resize((W, H), Image.BICUBIC), (LABEL+j*W, y))
            im.save(OUT/f'{name}.jpg', quality=90)
            print(name, flush=True)
    (OUT/'key.json').write_text(json.dumps(key, indent=1))


def score(name='counts.json'):
    key = json.loads((OUT/'key.json').read_text())
    counts = json.loads((OUT/name).read_text())
    if set(counts) != set(key):
        raise SystemExit(f'counts.json must list every panel: missing {sorted(set(key)-set(counts))}')
    rows = [dict(prompt=name.split('_')[0], condition=CONDITIONS[t], fail=int(label in counts[name]))
            for name, k in key.items() for label, t in k.items()]
    t = pd.DataFrame(rows).pivot_table(index='condition', columns='prompt', values='fail', aggfunc='sum')
    t = t[list(PROMPTS)].reindex(list(CONDITIONS.values()))
    t['합계 (32편 중)'] = t.sum(1)
    text = t.to_markdown()
    (OUT/f'score_{Path(name).stem}.md').write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('what', choices=['make', 'score'])
    a.add_argument('--counts', default='counts.json', help='counts file in the blind folder (score)')
    x = a.parse_args()
    make() if x.what == 'make' else score(x.counts)
