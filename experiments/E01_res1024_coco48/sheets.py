"""Image sheets of PROTOCOL_ko.md section 4: prompts coco_00, 04, ..., 44; per prompt three rows (Original, DAVE,
ours) x 8 seeds, 512 on the left and 1024 (full-resolution file, shown at the same thumbnail size) on the right.
Two prompts per sheet -> outputs/res1024/<model>/coco48/sheets/<model>_<k>.jpg.

  python experiments/E01_res1024_coco48/sheets.py sd35
"""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import MODES, NAMES, OUT                                         # noqa: E402

T, LABEL, HEAD, GAP = 176, 110, 34, 24
SEEDS = range(68000, 68008)
PIDS = [f'coco_{i:02d}' for i in range(0, 48, 4)]


def main(model):
    roots = {r: OUT/f'res{r}'/model/'coco48'/'generation' for r in (512, 1024)}
    out = OUT/'res1024'/model/'coco48'/'sheets'
    out.mkdir(parents=True, exist_ok=True)
    W = LABEL+2*8*T+GAP
    for k in range(0, len(PIDS), 2):
        pids = PIDS[k:k+2]
        sheet = Image.new('RGB', (W, len(pids)*(HEAD+3*T)+HEAD), 'white')
        d = ImageDraw.Draw(sheet)
        d.text((LABEL, 8), '512', fill='black')
        d.text((LABEL+8*T+GAP, 8), '1024', fill='black')
        y = HEAD
        for pid in pids:
            prompt = json.loads((roots[512]/pid/MODES[model]['orig']/'meta.json').read_text())['prompt']
            d.text((4, y+10), f'{pid}: {prompt}', fill='black')
            y += HEAD
            for role in ('orig', 'dave', 'ours'):
                d.text((4, y+T//2), NAMES[role], fill='black')
                for c, r in enumerate((512, 1024)):
                    for s, seed in enumerate(SEEDS):
                        p = roots[r]/pid/MODES[model][role]/f'seed{seed}.png'
                        if p.exists():
                            with Image.open(p) as im:
                                sheet.paste(im.convert('RGB').resize((T, T), Image.LANCZOS), (LABEL+c*(8*T+GAP)+s*T, y))
                y += T
        path = out/f'{model}_{k//2+1}.jpg'
        sheet.save(path, quality=88)
        print(path)


if __name__ == '__main__':
    main(sys.argv[1])
