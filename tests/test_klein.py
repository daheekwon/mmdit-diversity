"""CPU checks of the FLUX.2-klein additions (2026-11-05): splits, DAVE on a [text | image] stream, CFG step counting,
multi-step-set capture.  python -m pytest tests -q
"""
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import baselines, basis, config, data, splits                       # noqa: E402
from div.method import Controller                                             # noqa: E402

G = torch.Generator().manual_seed(0)    # local RNG: test_method.py draws from the global one seeded at import


def test_splits_match_value_diversity():
    sys.path.insert(0, str(splits.LEGACY.parent))
    from value_diversity.flux_transfer_plan import DEVELOPMENT_PROMPTS, DEVELOPMENT_SEEDS, CALIBRATION_SEEDS
    from value_diversity.combo_flux import RARE_PROMPTS
    from value_diversity import ho3_split, coco48_split
    rows, seeds = splits.split('dev')
    assert [(r['id'], r['prompt']) for r in rows] == [(p['id'], p['prompt']) for p in DEVELOPMENT_PROMPTS+RARE_PROMPTS]
    assert seeds == DEVELOPMENT_SEEDS
    rows, seeds = splits.split('dev8')
    assert [r['prompt'] for r in rows] == [p['prompt'] for p in DEVELOPMENT_PROMPTS] and seeds == CALIBRATION_SEEDS
    rows, seeds = splits.split('ho3')
    assert [(r['id'], r['prompt']) for r in rows] == [(p['id'], p['prompt']) for p in ho3_split.PROMPTS]
    assert seeds == list(range(66000, 66016))
    rows, seeds = splits.split('coco48')
    assert [(r['id'], r['prompt']) for r in rows] == [(p['id'], p['prompt']) for p in coco48_split.PROMPTS]
    assert seeds == coco48_split.SEEDS and len(rows) == 48


def test_split_datasets():
    d = OmegaConf.load(config.ROOT/'configs'/'data.yaml')
    for name, n in (('dev', 14), ('dev8', 8), ('ho3', 18), ('coco48', 48)):
        rows = data.prompts(d, name)
        assert len(rows) == n and all(len(r['seeds']) == 16 for r in rows)
        assert len({tuple(r['seeds']) for r in rows}) == 1                      # every prompt: the same seed list
    assert data.prompts(d, 'ho3', samples=8)[0]['seeds'] == list(range(66000, 66008))


class _Single(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb_mod=None):
        return hidden_states


def test_dave_image_rows_only():
    blk, ctrl = _Single(), type('C', (), dict(step=0, base=[]))()
    x = torch.randn(2, 30, 6, generator=G)+1
    h = baselines.dave_hook(blk, ctrl, .2, {0}, img_start=10)
    y = blk(x, None)
    h.remove()
    assert torch.equal(y[:, :10], x[:, :10])
    assert torch.allclose(y[:, 10:].mean(1), .2*x[:, 10:].mean(1), atol=1e-5)
    assert torch.allclose(y[:, 10:]-y[:, 10:].mean(1, keepdim=True), x[:, 10:]-x[:, 10:].mean(1, keepdim=True), atol=1e-5)


def test_cfg_step_counting():
    """Two transformer calls per step: both calls of a step see the same step and are edited."""
    c = object.__new__(Controller)
    c.model, c.steps, c.cfg_calls, c.cads = 'klein', 4, 2, None
    c.scope = {'v_edit': ({1, 2}, {0, 3})}
    c.step, c.n_calls, c.calls = -1, 0, []
    seen = []
    for _ in range(8):
        c._start(None, (), {})
        seen.append((c.step, sorted(c.active(2)), sorted(c.active(7))))
    assert [s for s, _, _ in seen] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [a for _, a, _ in seen] == [['v_edit'], ['v_edit'], [], [], [], [], ['v_edit'], ['v_edit']]
    import div.models as M
    old = M.N_BLOCKS.get('klein')
    assert len(c.calls) == 4 and c.expected_calls() == 2*2*2 and old == 25


def test_step_sets_equal_separate_captures():
    n, C = 4, 6
    xs = [(s, b, torch.randn(n, 5, C, generator=G)) for s in range(4) for b in (0, 1)]
    sets = basis.StepSets([0, 1], {'a': [0, 1], 'b': [0, 1, 2, 3]}, n)
    sets.ctrl = type('C', (), {})()
    sep = {'a': basis.Moments([0, 1], [0, 1], n), 'b': basis.Moments([0, 1], [0, 1, 2, 3], n)}
    for half in (0, 1):
        sets.half = half
        for m in sep.values():
            m.half = half
        for s, b, x in xs:
            sets.ctrl.step = s
            sets.add(b, x+half)
            for m in sep.values():
                if s in m.steps:
                    m.add(b, x+half)
    got = sets.state()
    for name, m in sep.items():
        ref = m.state()
        assert got[name].keys() == ref.keys()
        for k in ref:
            for f in ('D', 'C', 'H'):
                assert torch.equal(got[name][k][f], ref[k][f])
