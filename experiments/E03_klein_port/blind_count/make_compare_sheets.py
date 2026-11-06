"""Per-prompt comparison sheets (--s06: also ours steps 0-6, blocks 0-24, lambda 1.3/1.5/1.7, not eye-counted) (one prompt per figure): rows Original / DAVE steps 0-10 / ours lambda 1.3, 1.5, 1.7
(blocks 0-24, steps 0-10), all 16 seeds. Red frame = image counted as a failure (counts.csv + judge_only_verified.csv real=1)."""
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
H = Path(__file__).parent
EV = Path('/NHNHOME/workspace/mmdit-diversity/outputs/klein/eval/ho3/generation')
OUT = H.parent/'sheets'/'compare_final'
OUT.mkdir(exist_ok=True)
key = json.load(open(H/'key.json'))
ROWS = [('original', 'Original'), ('dave_a0.2~0-10', 'DAVE α0.2 블록 13, 스텝 0–10')] + \
       [(f'q_l{l}@0-24~0-10+pad_v0.25~0-10', f'우리 λ{l}, 블록 0–24, 스텝 0–10') for l in (1.3, 1.5, 1.7)]
import sys
S06 = '--s06' in sys.argv
if S06:
    ROWS = ROWS[:2] + [(f'q_l{l}@0-24~0-6+pad_v0.25~0-6', f'우리 λ{l}, 블록 0–24, 스텝 0–6') for l in (1.3, 1.5, 1.7)] + ROWS[2:]
    OUT = H.parent/'sheets'/'compare_final_s06'; OUT.mkdir(exist_ok=True)
COUNTED = {m for m, _ in ROWS if '~0-6' not in m}
c = pd.read_csv(H/'counts.csv'); c['mode'] = [key[p][r] for p, r in zip(c.pid, c.row)]
v = pd.read_csv(H/'judge_only_verified.csv'); v = v[v.real == 1]
fail = set(zip(c.pid, c['mode'], c.seed)) | set(zip(v.pid, v['mode'], v.seed))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
f_lab, f_title, f_seed = (ImageFont.truetype(FONT, n, index=1) for n in (26, 34, 22))   # index 1 = Noto Sans CJK KR
T, LAB, TOP, B = 200, 360, 70, 6
for pid in sorted(key):
    prompt = json.load(open(EV/pid/'original'/'meta.json'))['prompt']
    im = Image.new('RGB', (LAB + 16*T, TOP + len(ROWS)*T), 'white'); d = ImageDraw.Draw(im)
    d.text((10, 14), f'"{prompt}"  ({pid}, seed 66000–66015)   빨간 테두리 = 실패로 센 이미지' + ('  (스텝 0–6 행은 세지 않음)' if S06 else ''), fill='black', font=f_title)
    for r, (m, lab) in enumerate(ROWS):
        y = TOP + r*T
        nf = sum((pid, m, s) in fail for s in range(16))
        tail = f'\n실패 {nf}/16' if m in COUNTED else '\n실패: 세지 않음'
        d.multiline_text((10, y + T//2 - 34), lab.replace(', 블록', ',\n블록') + tail, fill='black', font=f_lab, spacing=4)
        for s in range(16):
            x = LAB + s*T
            if (pid, m, s) in fail:
                d.rectangle([x, y, x + T - 1, y + T - 1], fill=(220, 20, 20))
            im.paste(Image.open(EV/pid/m/f'seed{66000+s}.png').convert('RGB').resize((T - 2*B, T - 2*B)), (x + B, y + B))
            if r == 0:
                d.text((x + 10, y + 8), str(s), fill='yellow', font=f_seed, stroke_width=2, stroke_fill='black')
    im.save(OUT/f'{pid}.jpg', quality=88)
print(len(key), 'sheets ->', OUT)
