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


def test_splits_are_disjoint():
    """The splits of div/splits.py (made anew 2026-11-06) and the calibration set: sizes, unique ids, no shared prompt."""
    d = OmegaConf.load(config.ROOT/'configs'/'data.yaml')
    norm = lambda p: p.lower().rstrip('.')
    dev, seeds = splits.split('dev')
    assert len(dev) == 14 and seeds == list(range(50000, 50016))
    dev8, seeds = splits.split('dev8')
    assert dev8 == dev[:8] and seeds == list(range(40000, 40016))
    ho3, seeds = splits.split('ho3')
    assert len(ho3) == 18 and seeds == list(range(66000, 66016))
    calib = data.calibration_prompts(d)
    assert len(calib) == len(set(calib)) == 32
    groups = [{norm(r['prompt']) for r in dev}, {norm(r['prompt']) for r in ho3}, {norm(p) for p in calib}]
    assert [len(g) for g in groups] == [14, 18, 32]
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i+1:])
    ids = [r['id'] for r in dev+ho3]
    assert len(ids) == len(set(ids))


def test_split_datasets():
    d = OmegaConf.load(config.ROOT/'configs'/'data.yaml')
    splits_present = [('dev', 14), ('dev8', 8), ('ho3', 18)] + ([('coco48', 48)] if splits.COCO48.exists() else [])
    for name, n in splits_present:
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
