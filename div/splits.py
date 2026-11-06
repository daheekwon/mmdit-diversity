"""Fixed prompt splits with their own seed lists (2026-11-06, made anew in this repo; they replace the value_diversity/
splits of the earlier results, which are not available on this server, and are not the same prompts).

dev     8 general prompts (DEV_GENERAL: animal, object, scene, relation x 2) + 6 'A photo of a/an <rare>.' prompts
        (DEV_RARE: concepts outside the ImageNet-1k classes), seeds 50000-50015.
dev8    the 8 general dev prompts, seeds 40000-40015.
ho3     held-out 18 prompts (HO3: 6 categories x 3), seeds 66000-66015.
coco48  48 COCO val2017 captions (one per image), seeds 68000-68015, read from splits/coco48.json; that file is written
        by scripts/make_coco48.py, which keeps the images out of the COCO test and tune prompt files.
The V-edit calibration prompts are a separate set (splits/calib32.txt, data.calibration), disjoint from every split here
(tests/test_klein.py).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO48 = ROOT/'splits'/'coco48.json'
DEV_GENERAL = [('animal', ['A fox', 'A horse']), ('object', ['A lamp', 'A typewriter']),
               ('scene', ['A kitchen', 'A market street']),
               ('relation', ['A child flying a kite', 'Two people sharing a meal'])]
DEV_RARE = ['pangolin', 'okapi', 'theremin', 'astrolabe', 'quokka', 'samovar']
HO3 = [('animal', ['A deer in a forest', 'An owl', 'A cat sleeping on a sofa']),
       ('object', ['A wooden chair', 'An umbrella', 'A pair of sneakers']),
       ('scene', ['A beach at sunset', 'A mountain village', 'A subway station']),
       ('people', ['A chef cooking', 'A woman reading a book', 'A group of friends hiking']),
       ('food', ['A bowl of ramen', 'A slice of cake', 'A basket of fruit']),
       ('vehicle', ['A red car', 'A sailboat', 'A vintage train'])]
SEEDS = dict(dev=range(50000, 50016), dev8=range(40000, 40016), ho3=range(66000, 66016), coco48=range(68000, 68016))


def _article(w):
    return 'an' if w[0] in 'aeiou' else 'a'


def dev8():
    return [dict(id=f'dev_{g}_{i:02d}', prompt=t) for g, ts in DEV_GENERAL for i, t in enumerate(ts)]


def split(name):
    """(rows [{id, prompt}], seeds) of one split."""
    seeds = list(SEEDS[name]) if name in SEEDS else None
    if name == 'dev':
        rare = [dict(id='rare_'+w.replace(' ', '_'), prompt=f'A photo of {_article(w)} {w}.') for w in DEV_RARE]
        return dev8()+rare, seeds
    if name == 'dev8':
        return dev8(), seeds
    if name == 'ho3':
        return [dict(id=f'ho_{g}_{i:02d}', prompt=t) for g, ts in HO3 for i, t in enumerate(ts)], seeds
    if name == 'coco48':
        if not COCO48.exists():
            raise FileNotFoundError(f'{COCO48} not found: run scripts/make_coco48.py once the COCO test / tune files exist')
        rows = json.loads(COCO48.read_text())['prompts']
        return [dict(id=r['id'], prompt=r['prompt'], image_id=r['image_id']) for r in rows], seeds
    raise KeyError(name)
