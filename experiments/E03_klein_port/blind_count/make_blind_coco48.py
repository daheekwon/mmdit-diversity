"""Blind sheets for the COCO48 confirmation eye count (PROTOCOL_ko.md §11): per prompt, Original / DAVE steps 0-10 /
ours lambda 1.5 in a random row order A-C, seeds 0-7 and 8-15 on two sheets, 256 px thumbnails.
Key in coco48/key.json (not opened before coco48/counts.csv is complete)."""
import json, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
G = Path('/DATA0/dahee/diversity_migration/mmdit_diversity/outputs/klein/eval/coco48/generation')
OUT = Path(__file__).parent/'coco48'
MODES = ['original', 'dave_a0.2~0-10', 'q_l1.5@0-24~0-10+pad_v0.25~0-10']
T, LAB = 256, 40
rng = random.Random(20261107)
font = ImageFont.load_default(size=24)
(OUT/'sheets').mkdir(parents=True, exist_ok=True)
key, prompts = {}, {}
for pid in sorted(d.name for d in G.iterdir() if d.is_dir()):
    if not all((G/pid/m/'meta.json').exists() for m in MODES):
        print('incomplete, skipped:', pid); continue
    order = MODES[:]; rng.shuffle(order)
    key[pid] = dict(zip('ABC', order))
    prompts[pid] = json.load(open(G/pid/'original'/'meta.json'))['prompt']
    seeds = json.load(open(G/pid/'original'/'meta.json'))['seeds']
    for half in (0, 1):
        im = Image.new('RGB', (LAB + 8*T, 3*T + 36), 'white'); dr = ImageDraw.Draw(im)
        dr.text((4, 4), f'{pid} seeds {half*8}-{half*8+7}: {prompts[pid]}'[:160], fill='black', font=font)
        for r, m in enumerate(order):
            y = 36 + r*T
            dr.text((8, y + T//2 - 12), 'ABC'[r], fill='black', font=font)
            for c in range(8):
                s = half*8 + c
                im.paste(Image.open(G/pid/m/f'seed{seeds[s]}.png').convert('RGB').resize((T-4, T-4)), (LAB + c*T + 2, y + 2))
                dr.text((LAB + c*T + 6, y + 4), str(s), fill='yellow', font=font, stroke_width=2, stroke_fill='black')
        im.save(OUT/'sheets'/f'{pid}_{half}.jpg', quality=90)
json.dump(key, open(OUT/'key.json', 'w'), indent=1)
json.dump(prompts, open(OUT/'prompts.json', 'w'), indent=1)
print(len(key), 'prompts ->', OUT/'sheets')
