"""E06 (PROTOCOL_ko.md): prompts, seeds, condition names and folders.

Prompt set `e06` (50): coco_tune captions in file order rows 0-24 (tune_00..24), 12 hard-composition prompts (comp_01..12),
13 rare concepts 'A photo of a <w>.' (rare_<w>); seeds 71000-71007 (never used before).
Held-out `ho4` (24): value_diversity/ho4_split.py, seeds 67000-67015; 512 Original / DAVE are linked from
value_diversity/results/generalize/<model>/test/generation (same prompts, seeds and layout).
Condition names carry the applied value (lam for the V edit, factor for pad V).
"""
import importlib.util
import json
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
REPO = PKG.parent
OUT = PKG/'outputs'/'e06_boundary'
COCO_TUNE = Path('/DATA0/dahee/data/datasets/coco/captions_val2017_tune50_no_coco48.json')
RES = 512

COMP = [
    'A red cube on top of a blue sphere.',
    'Three yellow tulips in a glass vase next to two green apples.',
    'A cat sitting under a wooden table while a dog sleeps on a rug.',
    'A child holding an umbrella on the left and a bicycle leaning against a wall on the right.',
    'A teapot pouring tea into a cup on a saucer.',
    'Four people rowing a boat on a misty lake.',
    'A street lamp glowing over a snowy alley at night.',
    'A knight in armor reading a newspaper on a park bench.',
    'A violin lying on a piano next to a stack of sheet music.',
    'A small sailboat inside a glass bottle on a windowsill.',
    'A firefighter carrying a puppy down a ladder.',
    'A black king and a white queen chess piece on a checkered board.',
]
RARE = ['kinkajou', 'binturong', 'numbat', 'gerenuk', 'cassowary', 'hoatzin', 'potoo', 'samovar', 'orrery', 'zoetrope',
        'theremin', 'rambutan', 'romanesco broccoli']
LIGHTING = ['comp_06', 'comp_07']          # prompts that state the lighting (misty, night): dark-share check (4.1)

# VQA checks (yes/no, expected answer) for the composition prompts; rare prompts get one presence question.
CHECKS = dict(
    comp_01=['Is there a red cube?', 'Is there a blue sphere?', 'Is the cube on top of the sphere?'],
    comp_02=['Are there yellow tulips in a glass vase?', 'Are there exactly two green apples?'],
    comp_03=['Is a cat under a table?', 'Is a dog lying on a rug?'],
    comp_04=['Is a child holding an umbrella?', 'Is there a bicycle leaning against a wall?',
             'Is the child on the left side of the image?'],
    comp_05=['Is a teapot pouring tea into a cup?', 'Is the cup on a saucer?'],
    comp_06=['Are people rowing a boat?', 'Are there exactly four people in the boat?'],
    comp_07=['Is there a glowing street lamp?', 'Is it night?', 'Is there snow?'],
    comp_08=['Is there a knight in armor?', 'Is the knight reading a newspaper?', 'Is the knight sitting on a bench?'],
    comp_09=['Is there a violin lying on a piano?', 'Is there sheet music?'],
    comp_10=['Is there a sailboat inside a glass bottle?', 'Is the bottle on a windowsill?'],
    comp_11=['Is a firefighter carrying a puppy?', 'Is there a ladder?'],
    comp_12=['Is there a black king chess piece?', 'Is there a white queen chess piece?', 'Is there a checkered board?'],
)
# Object nouns for the CLIPSeg object-size measure (attribute-free).
OBJECTS = dict(
    tune_00='person', tune_01='cake|plate', tune_02='person|skis', tune_03='person|baseball bat', tune_04='sheep',
    tune_05='man|snowboard', tune_06='kitten|bicycle', tune_07='woman|elephant', tune_08='wheelbarrow|surfboard',
    tune_09='cake pan|stove', tune_10='person|table', tune_11='man|tie', tune_12='sofa|table', tune_13='man',
    tune_14='man|tennis racket', tune_15='microwave', tune_16='menu', tune_17='train', tune_18='animal',
    tune_19='man|tie', tune_20='woman|pizza', tune_21='bicycle|bed|chair', tune_22='pizza|salad',
    tune_23='cat|television', tune_24='cabinet|vase',
    comp_01='cube|sphere', comp_02='tulip|vase|apple', comp_03='cat|table|dog', comp_04='child|umbrella|bicycle',
    comp_05='teapot|cup', comp_06='boat|person', comp_07='street lamp', comp_08='knight|bench', comp_09='violin|piano',
    comp_10='bottle|sailboat', comp_11='firefighter|puppy|ladder', comp_12='chess piece|board')

SEEDS = dict(e06=list(range(71000, 71008)), ho4=list(range(67000, 67016)))
ORIG = dict(sd35='original', flux='original')
HO4_LINK = dict(sd35={'original': 'original', 'dave_a0.5': 'dave_a0.5'},
                flux={'b8_original': 'b8_original', 'dave_a0.2': 'dave_a0.2'})
HO4_REF = dict(sd35='original', flux='b8_original')

# condition -> (preset, overrides); model.resolution=512 is added for every condition.
_V = dict(sd35=['1.04', '1.07', '1.1', '1.14', '1.17', '1.2', '1.3'],
          flux=['1.2', '1.43', '1.6', '1.72', '2.01', '2.3', '2.58'])     # 1.17: §10; 1.04 / 1.1, 1.2 / 1.6: §12
E1_LOW = dict(sd35=['v_l1.04', 'v_l1.1'], flux=['v_l1.2', 'v_l1.6'])
_PAD = ['0.5', '0.25', '0.125', '0']


COMBO_LOW = dict(sd35='1.07', flux='1.43')


def modes(model):
    m = {'original': ('original', [])}
    m.update({f'v_l{v}': ('v_edit', [f'v_edit.lam={v}']) for v in _V[model]})
    m['v_l1.14+padv0.25' if model == 'sd35' else 'v_l1.72+padv0.25'] = ('ours', [])
    low = COMBO_LOW[model]                     # §13: combination path (pad 1/4 fixed), one point below the final lambda
    m[f'v_l{low}+padv0.25'] = ('ours', [f'v_edit.lam={low}'])
    m.update({f'padv{f}': ('pad_v', [f'pad_v.factor={f}']) for f in _PAD})
    m['dave_a0.5' if model == 'sd35' else 'dave_a0.2'] = ('dave', [])
    m['cads0.25'] = ('cads', [])
    return m


E06_MODES = dict(sd35=list(modes('sd35')), flux=list(modes('flux')))

# ---------------------------------------------------------------- stage 2 (PROTOCOL_ko.md §9)
SIGNAL_STEPS = dict(sd35=(0, 1, 11, 12), flux=(0, 1, 5, 6))   # first, second, last edited, first unedited step
SIGNAL_SEEDS = [71000, 71001]
E3_V = dict(sd35=['v_l1.14', 'v_l1.2'], flux=['v_l1.72', 'v_l2.01'])   # E3 image-V points (§3; moved if E1 shows a knee)
E3_PAD = [f'padv{f}' for f in _PAD]       # E3 pad point = the E2 onset s (§3, §11), chosen with add_stage2.py --e3 --pad
E3_DAVE = dict(sd35='dave_a0.5', flux='dave_a0.2')   # §17 (added after the SD3.5 E3 result): DAVE standard + same blocks
# §20 E4 guard (run only where E3 found a block that cut only the cost: FLUX pad 1/4, both blocks): the combination path
# (pad 1/4 fixed) at the three E4 lambdas, each with the block; the unguarded upper point is the extra comparison point (§13).
E4_LAM = dict(flux=['1.43', '1.72', '2.01'])
E4_FIX = dict(flux=['it', 'txt'])


def e4_names(model):
    """(unguarded points to generate, guarded conditions) of E4."""
    lam = E4_LAM.get(model, [])
    return ([f'v_l{lam[-1]}+padv0.25'] if lam else [],
            [f'v_l{v}+padv0.25+fix{x}' for v in lam for x in E4_FIX[model]])


def random_basis(model):
    return OUT/'bases'/f'{model}_random_k256.pt'


def write_json(model):
    return OUT/model/'write_measure.json'


def random_mode(model):
    """(name, preset, overrides) of the random-direction condition, once the write measure has fixed lam_r."""
    p = write_json(model)
    if not p.exists():
        return None
    lam = json.loads(p.read_text())['lam_random']
    return f'rnd_l{lam:.3f}', 'v_edit', [f'v_edit.basis={random_basis(model)}', f'v_edit.lam={lam:.3f}']


def all_modes(model):
    """Stage-1 table + the random condition (if measured) + the E3 conditions: name -> (preset, overrides, twin)."""
    t = {k: (p, o, ()) for k, (p, o) in modes(model).items()}
    rm = random_mode(model)
    if rm:
        t[rm[0]] = (rm[1], rm[2], ())
    for base in E3_V[model]+E3_PAD+[E3_DAVE[model]]:
        p, o = modes(model)[base]
        t[f'{base}+fixit'] = (p, o, ('it',))
        t[f'{base}+fixtxt'] = (p, o, ('txt',))
    for base in E3_V[model]:                       # DESIGN_e3_split_ko.md A: image queries read the twin's text K / V
        p, o = modes(model)[base]
        for suf, twin in SPLIT.items():
            t[f'{base}+{suf}'] = (p, o, twin)
    m = modes(model)
    for v in E4_LAM.get(model, []):            # §20 E4: combination points (pad 1/4 fixed) +/- the E3 block
        base = f'v_l{v}+padv0.25'
        p, o = m[base] if base in m else ('ours', [f'v_edit.lam={v}'])
        t.setdefault(base, (p, o, ()))
        for x in E4_FIX[model]:
            t[f'{base}+fix{x}'] = (p, o, (x,))
    for v in VROWS[model]['current']:          # DESIGN_vrows_ko.md: extra all-rows combination point
        t.setdefault(f'v_l{v}+padv0.25', ('ours', [f'v_edit.lam={v}'], ()))
    for name, (p, o) in vrows_modes(model).items():
        t[name] = (p, o, ())
    return t


# DESIGN_e3_split_ko.md A (added after the SD3.5 E3 result). Names avoid '+fix' so the default E3 mode list is unchanged.
SPLIT = {'readkv': ('tk', 'tv'), 'readv': ('tv',), 'readk': ('tk',), 'readwkv': ('wk', 'wv')}
SPLIT_LAM = dict(sd35='v_l1.2', flux='v_l2.01')

# DESIGN_vrows_ko.md (added after the E06 strict labels): the V edit read by image-query rows only (v_edit.rows=image).
# 'vimg_l<lam>' = image-rows V edit alone, 'vimg_l<lam>+padv0.25' = with pad V 1/4; names avoid 'v_l' so the E1 lists of
# analyze*.py stay the all-rows curve. 'current' = all-rows combination points generated for the comparison curve.
VROWS = dict(sd35=dict(v=['1.14', '1.2', '1.3'], combo=['1.14', '1.2'], current=['1.2']),
             flux=dict(v=['1.72', '2.01', '2.58'], combo=['1.72', '2.01'], current=[]))


def vrows_modes(model):
    d = VROWS[model]
    t = {f'vimg_l{v}': ('v_edit', [f'v_edit.lam={v}', 'v_edit.rows=image']) for v in d['v']}
    t.update({f'vimg_l{v}+padv0.25': ('ours', [f'v_edit.lam={v}', 'v_edit.rows=image']) for v in d['combo']})
    return t


def vrows_new(model):
    """Conditions to generate: the variant points and the extra all-rows combination points."""
    return list(vrows_modes(model))+[f'v_l{v}+padv0.25' for v in VROWS[model]['current']]


def signal_modes(model):
    names = [k for k in modes(model) if not k.startswith(('dave', 'cads'))]
    rm = random_mode(model)
    return names+([rm[0]] if rm else [])
HO4_MODES = dict(sd35=['v_l1.14', 'v_l1.2', 'padv0.25', 'padv0.125', 'cads0.25'],
                 flux=['v_l1.72', 'v_l2.01', 'padv0.25', 'padv0.125', 'cads0.25'])


def prompts(split):
    if split == 'e06':
        d = json.loads(COCO_TUNE.read_text())
        cap = {a['image_id']: a['caption'].strip() for a in d['annotations']}
        rows = [dict(id=f'tune_{i:02d}', prompt=cap[im['id']], image_id=im['id'], group='coco')
                for i, im in enumerate(d['images'][:25])]
        rows += [dict(id=f'comp_{i+1:02d}', prompt=p, group='comp') for i, p in enumerate(COMP)]
        rows += [dict(id='rare_'+w.replace(' ', '_'), prompt=f'A photo of a {w}.', group='rare') for w in RARE]
        return rows
    if split == 'ho4':
        spec = importlib.util.spec_from_file_location('ho4_split', REPO/'value_diversity'/'ho4_split.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return [dict(id=p['id'], prompt=p['prompt'], group='rare' if p['id'].startswith('rare') else p['id'].split('_')[1])
                for p in mod.PROMPTS]
    raise KeyError(split)


def gen_root(model, split):
    return OUT/model/split/'generation'
