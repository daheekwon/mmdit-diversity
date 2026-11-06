"""Config: configs/<model>.yaml (model settings, component defaults, presets) + configs/data.yaml + CLI overrides."""
import re
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ('v_edit', 'pad_v', 'head', 'dave', 'cads')


def load(model, preset='original', overrides=()):
    """overrides: dotlist such as ['v_edit.strength=0.35', 'model.batch_size=8']; component keys need no 'method.'."""
    spec = OmegaConf.load(ROOT/'configs'/f'{model}.yaml')
    if preset not in spec.presets:
        raise KeyError(f'unknown preset {preset!r}; presets: {list(spec.presets)}')
    method = OmegaConf.merge(spec.method, spec.presets[preset] or {})
    cfg = OmegaConf.create(dict(model=spec.model, method=method, data=OmegaConf.load(ROOT/'configs'/'data.yaml'),
                                preset=preset, overrides=list(overrides)))
    dot = [('method.'+o if o.split('.')[0] in COMPONENTS else o) for o in overrides]
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot))
    for c in COMPONENTS:                       # an override of an unknown field is a typo, not a new setting
        extra = set(cfg.method[c]) - set(spec.method[c])
        if extra:
            raise KeyError(f'unknown {c} fields: {sorted(extra)}')
    v = cfg.method.v_edit                      # lam would silently win over a strength override
    if v.lam is not None and v.strength is not None:
        raise ValueError('set v_edit.lam or v_edit.strength, not both (override the other with =null)')
    return cfg


def tag(cfg):
    """Folder name of a run: preset plus its overrides, e.g. ours__v_edit.strength=0.35."""
    parts = [cfg.preset] + sorted(cfg.overrides)
    return re.sub(r'[^A-Za-z0-9_.=,+-]', '_', '__'.join(parts))


def parse_range(spec, n):
    """'all' | '6-23' | '0,2,5-7' -> sorted list of ints in [0, n)."""
    if str(spec) == 'all':
        return list(range(n))
    out = set()
    for part in str(spec).split(','):
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo)+1))
    if not out or min(out) < 0 or max(out) >= n:
        raise ValueError(f'range {spec!r} outside 0..{n-1}')
    return sorted(out)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT/p
