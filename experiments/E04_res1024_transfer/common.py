"""Paths and condition names of PROTOCOL_ko.md (final method at 512 and 1024, COCO 48 x seeds 68000-68007)."""
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
OUT = PKG/'outputs'/'e04_final'
E01 = PKG/'outputs'                       # E01 roots: outputs/res<res>/<model>/coco48/{generation,eval512}
SEEDS = range(68000, 68008)
FINAL = dict(sd35='v_l1.14@0-23+padv0.25~0-11', flux='v_l1.72@19-37+padv0.25~0-5')
ORIG = dict(sd35='original', flux='b8_original')
DAVE = dict(sd35='dave_a0.5', flux='dave_a0.2')


def gen_root(model, res):
    return OUT/f'res{res}'/model/'coco48'/'generation'


def eval_root(model, res):
    """Folder the evaluation reads: the 512 images themselves, or the 1024 images resized to 512."""
    return gen_root(model, res) if res == 512 else gen_root(model, res).parent/'eval512'


def e01_root(model, res, kind='generation'):
    return E01/f'res{res}'/model/'coco48'/kind
