"""The prompt splits of value_diversity/ (dev, ho3, COCO 48) with their fixed seeds, so a new model can be evaluated on
exactly the prompts and seeds of the SD3.5 / FLUX.1 generalize program (value_diversity/results/generalize/).

dev     8 general dev prompts (value_diversity/flux_transfer_plan.py DEVELOPMENT_PROMPTS) + 6 'A photo of a <rare>.'
        prompts (combo_flux.py RARE), seeds 50000-50015 (DEVELOPMENT_SEEDS).
dev8    the 8 general dev prompts only: the V-edit basis calibration prompts of every earlier result, with the
        calibration seeds 40000-40015 (CALIBRATION_SEEDS).
ho3     value_diversity/ho3_split.py (18 prompts), seeds 66000-66015 (the generalize tune runs used 16).
coco48  value_diversity/coco48_split.py (48 captions), seeds 68000-68015.
ho3_split.py and coco48_split.py hold plain literals only; they are executed from their files, not imported, so the
value_diversity package is not loaded. tests/test_method.py checks dev / dev8 against value_diversity itself.
"""
from pathlib import Path

LEGACY = next(p/'value_diversity' for p in Path(__file__).resolve().parents if (p/'value_diversity').is_dir())
DEV_GENERAL = [('animal', ['A red panda', 'A goat']), ('object', ['A kettle', 'A camera']),
               ('scene', ['A library', 'A courtyard']), ('relation', ['A person watering plants', 'People playing a board game'])]
DEV_RARE = ['rapeseed', 'daisy', 'digital watch', 'porcupine', 'yurt', 'black stork']


def _literals(name):
    ns = {}
    exec(compile((LEGACY/name).read_text(), str(LEGACY/name), 'exec'), ns)
    return ns


def dev8():
    return [dict(id=f'dev_{g}_{i:02d}', prompt=t) for g, ts in DEV_GENERAL for i, t in enumerate(ts)]


def split(name):
    """(rows [{id, prompt}], seeds) of one split."""
    if name == 'dev':
        rare = [dict(id='rare_'+w.replace(' ', '_'), prompt=f'A photo of a {w}.') for w in DEV_RARE]
        return dev8()+rare, list(range(50000, 50016))
    if name == 'dev8':
        return dev8(), list(range(40000, 40016))
    if name == 'ho3':
        ns = _literals('ho3_split.py')
        return [dict(id=p['id'], prompt=p['prompt']) for p in ns['PROMPTS']], list(range(66000, 66016))
    if name == 'coco48':
        ns = _literals('coco48_split.py')
        return [dict(id=p['id'], prompt=p['prompt'], image_id=p['image_id']) for p in ns['PROMPTS']], ns['SEEDS']
    raise KeyError(name)
