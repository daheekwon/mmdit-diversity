"""Blind sheets for the ho3 eye count (CRITERIA_ko.md): per prompt, the 5 conditions in a random row order A-E,
seeds 0-7 and 8-15 on two sheets. Key in key.json (not opened before counts.csv is complete)."""
import json, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
G = Path('/NHNHOME/workspace/mmdit-diversity/outputs/klein/eval/ho3/generation')
OUT = Path(__file__).parent
MODES = ['original', 'dave_a0.2~0-10'] + [f'q_l{l}@0-24~0-10+pad_v0.25~0-10' for l in (1.3, 1.5, 1.7)]
T, LAB = 192, 40
rng = random.Random(20261106)
font = ImageFont.load_default(size=22) if hasattr(ImageFont, 'load_default') else None
key = {}
(OUT/'sheets').mkdir(exist_ok=True)
for pid in sorted(d.name for d in G.iterdir() if d.is_dir()):
    order = MODES[:]; rng.shuffle(order)
    key[pid] = dict(zip('ABCDE', order))
    for half in (0, 1):
        im = Image.new('RGB', (LAB + 8*T, 5*T + 30), 'white'); dr = ImageDraw.Draw(im)
        dr.text((4, 4), f'{pid} seeds {half*8}-{half*8+7}', fill='black', font=font)
        for r, m in enumerate(order):
            y = 30 + r*T
            dr.text((8, y + T//2 - 12), 'ABCDE'[r], fill='black', font=font)
            for c in range(8):
                s = half*8 + c
                p = G/pid/m/f'seed{66000+s}.png'
                im.paste(Image.open(p).convert('RGB').resize((T-4, T-4)), (LAB + c*T + 2, y + 2))
                dr.text((LAB + c*T + 6, y + 4), str(s), fill='yellow', font=font, stroke_width=2, stroke_fill='black')
        im.save(OUT/'sheets'/f'{pid}_{half}.jpg', quality=88)
json.dump(key, open(OUT/'key.json', 'w'), indent=1)
print(len(key), 'prompts')
