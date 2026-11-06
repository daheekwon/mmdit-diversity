"""CPU checks of the method, baselines, basis fit and metrics against brute-force references.

  python -m pytest tests -q
GPU checks against the earlier value_diversity/ images: tests/parity.py.
"""
import math
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import baselines, basis, config, data, metrics                     # noqa: E402
from div.method import Controller, frame_mean, head_delta                  # noqa: E402

torch.manual_seed(0)


def fake_ctrl(factor=1., beta=0., n_pad=5, n_txt=12):
    c = object.__new__(Controller)
    c.cfg = OmegaConf.create(dict(method=dict(pad_v=dict(factor=factor), head=dict(beta=beta))))
    c.pad_index = torch.arange(n_txt-n_pad, n_txt)
    return c


def qkv(B=2, H=3, S=40, D=8):
    return [torch.randn(B, H, S, D, dtype=torch.float64) for _ in range(3)]


def reference(q, k, v, img, txt, pos, f, beta):
    """Explicit softmax; image rows read V with pad rows x f, minus beta (1 - sbar) A_II V_I."""
    a = (q@k.transpose(-1, -2)/math.sqrt(q.shape[-1])).softmax(-1)
    vp = v.clone()
    vp[:, :, pos] *= f
    out = a@v
    out[:, :, img] = a[:, :, img]@vp
    aii = a[:, :, img][..., img]
    n = aii.shape[2]
    m, e = aii.sum(2)/n, aii.square().sum(2)/n
    sbar = (1-m**2/e).clamp_min(0).mean(-1)
    out[:, :, img] -= beta*(1-sbar)[..., None, None]*(aii@v[:, :, img])
    return out


@pytest.mark.parametrize('img_first', [True, False])
@pytest.mark.parametrize('f,beta', [(1., 0.), (.125, 0.), (.25, .5), (1., .85)])
def test_attend_matches_reference(img_first, f, beta):
    q, k, v = qkv()
    n_img, n_txt = 28, 12
    img, txt = (slice(0, n_img), slice(n_img, None)) if img_first else (slice(n_txt, None), slice(0, n_txt))
    c = fake_ctrl(f, beta, n_txt=n_txt)
    act = {'pad_v', 'head'}
    out = c.attend(q, k, v, img, txt, act)
    pos = txt.start+c.pad_index
    tol = 1e-5 if beta else 1e-10                                   # the head term is computed in float32
    assert torch.allclose(out, reference(q, k, v, img, txt, pos, f, beta), atol=tol)


def test_attend_without_edits_is_plain_attention():
    q, k, v = qkv()
    assert torch.equal(fake_ctrl().attend(q, k, v, slice(0, 28), slice(28, None), set()),
                       F.scaled_dot_product_attention(q, k, v))


def test_head_delta_chunking():
    q, k, v = qkv(S=300)
    img = slice(0, 280)
    import div.method as M
    full = head_delta(q[:, :, img], k, v, img, .7)
    M.HEAD_CHUNK, old = 7, M.HEAD_CHUNK
    try:
        assert torch.allclose(head_delta(q[:, :, img], k, v, img, .7), full, atol=1e-6)
    finally:
        M.HEAD_CHUNK = old


def test_image_value_edit():
    c = object.__new__(Controller)
    C = 16
    Q = torch.linalg.qr(torch.randn(C, 4))[0]
    c.basis_path, c.capture = 'x', None
    import div.method as M
    M._Q[('x', 3, torch.device('cpu'))] = Q
    v, b = torch.randn(2, 10, C), torch.randn(C)
    c.lam = 1.
    assert torch.equal(c.image_value(3, v, b, {'v_edit'}), v)
    c.lam = 1.7
    assert torch.allclose(c.image_value(3, v, b, {'v_edit'}), v+.7*((v-b)@Q@Q.T), atol=1e-6)
    assert torch.equal(c.image_value(3, v, b, set()), v)


def test_image_value_bit_identical_to_out_of_place():
    """The in-place float32 edit equals the earlier out-of-place float32 edit bit for bit (bf16 / fp16 / fp32 input,
    with and without bias), leaves v unchanged, and hands the capture the unedited bias-free V."""
    c = object.__new__(Controller)
    C = 64
    Q = torch.linalg.qr(torch.randn(C, 8))[0]
    c.basis_path, c.lam, c.time_mean = 'x', 1.72, False
    import div.method as M
    M._Q[('x', 3, torch.device('cpu'))] = Q

    class Keep:
        def add(self, block, x):
            self.x = x                                  # no copy: a later in-place change would show
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        v, b = (3*torch.randn(2, 50, C)).to(dt), torch.randn(C).to(dt)
        keep = v.clone()
        for bias, x in ((b, v.float()-b.float()), (None, v.float())):
            c.capture = None
            assert torch.equal(c.image_value(3, v, bias, {'v_edit'}), (v.float()+(c.lam-1)*((x@Q)@Q.T)).to(dt))
            c.capture = Keep()
            c.image_value(3, v, bias, {'capture', 'v_edit'})
            assert torch.equal(c.capture.x, x)
        assert torch.equal(v, keep)


def test_time_mean_edit():
    """Time-mean V edit against a loop over positions; one frame = the plain edit; config presets."""
    c = object.__new__(Controller)
    C, P, Fr = 16, 5, 3
    Q = torch.linalg.qr(torch.randn(C, 4))[0]
    c.basis_path, c.capture, c.lam = 'x', None, 1.7
    import div.method as M
    M._Q[('x', 3, torch.device('cpu'))] = Q
    v, b = torch.randn(2, Fr*P, C), torch.randn(C)
    d = (v-b)@Q@Q.T
    want = v.clone()
    for p in range(P):
        m = torch.stack([d[:, f*P+p] for f in range(Fr)]).mean(0)
        for f in range(Fr):
            want[:, f*P+p] += .7*m
    c.time_mean, c.frame_tokens = True, P
    assert torch.allclose(c.image_value(3, v, b, {'v_edit'}), want, atol=1e-6)
    one = v[:, :P]
    c.frame_tokens = P
    assert torch.allclose(c.image_value(3, one, b, {'v_edit'}), one+.7*((one-b)@Q@Q.T), atol=1e-6)
    with pytest.raises(ValueError):
        frame_mean(d, 4)
    t = config.load('cogvideox', 'ours_time_mean')
    assert t.method.v_edit.time_mean and t.method.pad_v.enabled and t.method.v_edit.strength == 0.22
    assert not config.load('cogvideox', 'ours').method.v_edit.time_mean
    with pytest.raises(KeyError):
        config.load('sd35', 'ours', ['v_edit.time_mean=true'])


def test_parse_range_and_config():
    assert config.parse_range('all', 4) == [0, 1, 2, 3]
    assert config.parse_range('0,2,5-7', 24) == [0, 2, 5, 6, 7]
    with pytest.raises(ValueError):
        config.parse_range('20-30', 24)
    cfg = config.load('sd35', 'ours', ['v_edit.lam=1.2', 'pad_v.blocks=0-5'])
    assert cfg.method.v_edit.enabled and cfg.method.pad_v.enabled and not cfg.method.dave.enabled
    assert cfg.method.v_edit.lam == 1.2 and cfg.method.pad_v.blocks == '0-5'
    assert config.tag(cfg) == 'ours__pad_v.blocks=0-5__v_edit.lam=1.2'
    with pytest.raises(KeyError):
        config.load('flux', 'ours', ['v_edit.strenght=0.3'])
    with pytest.raises(ValueError):                      # the default lam would hide a strength override
        config.load('flux', 'ours', ['v_edit.strength=0.5'])
    assert config.load('flux', 'ours', ['v_edit.lam=null', 'v_edit.strength=0.5']).method.v_edit.lam is None


def test_defaults_one_lam():
    """SD3.5 / FLUX defaults (2026-11-05): one lam on every edited block, pad V on the V-edit steps, and the SD3.5
    all-block basis equal to the 6-23 basis on 6-23."""
    for model, lam, blocks, steps in (('sd35', 1.14, 'all', '0-11'), ('flux', 1.72, '19-37', '0-5')):
        m = config.load(model, 'ours').method
        assert (m.v_edit.lam, m.v_edit.strength, m.v_edit.blocks, m.v_edit.steps) == (lam, None, blocks, steps)
        assert m.pad_v.blocks == 'all' and m.pad_v.steps == m.v_edit.steps
    a, b = (torch.load(config.ROOT/'bases'/f'sd35_legacy_dev8{s}_k256.pt', weights_only=True) for s in ('', '_b0to23'))
    assert b['blocks'] == list(range(24)) and all(torch.equal(a['Q'][k], b['Q'][k]) and a['unit'][k] == b['unit'][k] for k in a['Q'])


class _Block(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None, temb=None):
        return hidden_states


def test_dave_scales_token_mean():
    blk, ctrl = _Block(), type('C', (), dict(step=0, base=[]))()
    x = torch.randn(3, 50, 8)+2
    h = baselines.dave_hook(blk, ctrl, .2, {0})
    y = blk(x)
    h.remove()
    assert torch.allclose(y.mean(1), .2*x.mean(1), atol=1e-5)
    assert torch.allclose(y-y.mean(1, keepdim=True), x-x.mean(1, keepdim=True), atol=1e-5)


@pytest.mark.parametrize('groups', [1, 2])
def test_cads_is_per_image(groups):
    seeds, step = [11, 12, 13], 1
    cfg = OmegaConf.create(dict(noise_scale=.25, tau1=.6, tau2=.9, psi=1., rescale=True))
    text = torch.randn(groups*len(seeds), 7, 5)
    run = lambda sd, t: baselines.CADS(cfg, sd, 28)(_Block(), (), dict(hidden_states=None, encoder_hidden_states=t), 0)[0][1]
    batch = run(seeds, text)
    for j, s in enumerate(seeds):
        rows = list(range(j, len(text), len(seeds)))
        single = run([s], text[rows])
        assert torch.equal(batch[rows], single)
    assert baselines.cads_schedule(1-27/28, .6, .9) == 1.


def test_basis_fit_finds_seed_subspace():
    C, k, n = 12, 3, 8
    dev_dirs, common_dirs = torch.eye(C)[:, :k], torch.eye(C)[:, k:2*k]
    mom = basis.Moments([0], [0], n)
    for p in range(6):
        mom.half = p % 2
        base = torch.randn(1, 20, k)@common_dirs.T*3
        x = base+torch.randn(n, 20, k)@dev_dirs.T+.05*torch.randn(n, 20, C)
        mom.add(0, x)
    b = basis.fit(mom.state(), {0: torch.zeros(C)}, 'sd35', k=k)
    Q = b['Q'][0].double()
    assert torch.allclose(Q.T@Q, torch.eye(k, dtype=Q.dtype), atol=1e-5)
    assert (Q.T@dev_dirs.double()).norm()**2 > k-.05
    assert b['check'][0]['selectivity_ab'] > 10 and 0 < b['unit'][0] < 1


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 6))
    assert abs(metrics.fid(a, a)) < 1e-6
    r = metrics.prdc(a, a)
    assert r['precision'] == r['recall'] == r['coverage'] == 1.
    assert metrics.vendi(np.ones((5, 4))) == pytest.approx(1.)
    assert metrics.vendi(np.eye(5)) == pytest.approx(5.)


def test_imagenet_prompts_and_real_set():
    d = OmegaConf.load(config.ROOT/'configs'/'data.yaml')
    rows = data.prompts(d, 'imagenet')
    assert len(rows) == 1000 and rows[0]['wnid'] == 'n01440764' and rows[0]['prompt'] == 'a photo of a tench'
    seeds = [s for r in rows for s in r['seeds']]
    assert len(seeds) == len(set(seeds)) == 10000
    sub = data.prompts(d, 'imagenet', n_prompts=50)
    assert len(sub) == 50 and all(r['seeds'] == rows[r['index']]['seeds'] for r in sub)
    real = data.real_images(d, 'imagenet')
    assert len(real) == 10000 and all(p.exists() for p in real[::997])


def test_coco_prompts():
    d = OmegaConf.load(config.ROOT/'configs'/'data.yaml')
    rows = data.prompts(d, 'coco')
    assert len(rows) == 500 and len({r['id'] for r in rows}) == 500 and len({r['image_id'] for r in rows}) == 500
    assert rows[0]['id'] == 'coco_738712' and rows[0]['prompt'] == 'A person on a skateboard does a flip in a parking lot.'
    assert all(r['prompt'] == r['prompt'].strip() and r['prompt'] for r in rows)
    seeds = [s for r in rows for s in r['seeds']]
    assert len(seeds) == len(set(seeds)) == 5000 and min(seeds) == 1000000
    sub = data.prompts(d, 'coco', n_prompts=50)
    assert len(sub) == 50 and all(r['seeds'] == rows[r['index']]['seeds'] for r in sub)
    real = data.real_images(d, 'coco')
    assert len(real) == 5000


def test_tune_sets_are_disjoint_from_test():
    d = OmegaConf.load(config.ROOT/'configs'/'data.yaml')
    seeds = {k: {s for r in data.prompts(d, k) for s in r['seeds']} for k in data.DATASETS
             if data.base(d, k) != 'split'}           # the value_diversity splits keep their legacy seed lists
    assert all(not (seeds[a] & seeds[b]) for a in seeds for b in seeds if a < b)
    tune = data.prompts(d, 'imagenet_tune')
    assert len(tune) == 50 and {r['id'] for r in tune} == {r['id'] for r in data.prompts(d, 'imagenet', n_prompts=50)}
    coco_test = {r['image_id'] for r in data.prompts(d, 'coco')}
    coco_tune = data.prompts(d, 'coco_tune')
    assert len(coco_tune) == 50 and not ({r['image_id'] for r in coco_tune} & coco_test)
    assert data.base(d, 'coco_tune') == 'coco' and len(data.real_images(d, 'imagenet_tune')) == 10000
    c48 = {r['image_id'] for r in data.prompts(d, 'coco48')}     # the split where the operating lam was chosen (E01)
    assert not (c48 & coco_test) and not (c48 & {r['image_id'] for r in coco_tune})


def test_subclass_attend_is_routed():
    class Measure(Controller):
        def attend(self, q, k, v, img, txt, act):
            return super().attend(q, k, v, img, txt, act)
    plain, measure = object.__new__(Controller), object.__new__(Measure)
    assert not plain.wants_attend({'capture'}) and not plain.wants_attend({'v_edit'})
    assert plain.wants_attend({'pad_v'}) and plain.wants_attend({'head', 'v_edit'})
    assert measure.wants_attend({'capture'})


@pytest.mark.parametrize('img_first', [True, False])
@pytest.mark.parametrize('f', [1., .25])
def test_v_rows_image(img_first, f):
    """v_edit.rows=image (DESIGN_vrows_ko.md): image-query rows read the edited image V (and pad_v), text-query rows read
    the unedited image V; image_value's output equals rows=all; the kept V is consumed once; subclasses are refused."""
    import div.method as M
    B, H, D, n_img, n_txt = 2, 3, 8, 28, 12
    C = H*D
    Q = torch.linalg.qr(torch.randn(C, 4, dtype=torch.float64))[0].float()
    M._Q[('x', 5, torch.device('cpu'))] = Q
    c = fake_ctrl(f, 0., n_txt=n_txt)
    c.basis_path, c.lam, c.capture, c.v_rows = 'x', 1.7, None, 'image'
    v_img, b = torch.randn(B, n_img, C), torch.randn(C)
    ref_edit = (v_img.float()+.7*((v_img-b)@Q@Q.T)).to(v_img.dtype)
    edited = c.image_value(5, v_img, b, {'v_edit'})
    assert torch.equal(edited, ref_edit) and c._v_txt is v_img
    assert c.wants_attend({'v_edit'}) and not fake_ctrl().wants_attend({'v_edit'})
    img, txt = (slice(0, n_img), slice(n_img, None)) if img_first else (slice(n_txt, None), slice(0, n_txt))
    heads = lambda x: x.reshape(B, -1, H, D).transpose(1, 2)
    q, k = (torch.randn(B, H, n_img+n_txt, D) for _ in range(2))
    tv = torch.randn(B, H, n_txt, D)
    cat = lambda vi: torch.cat([vi, tv], 2) if img_first else torch.cat([tv, vi], 2)
    v_edit, v_orig = cat(heads(edited)), cat(heads(v_img))
    act = {'v_edit', 'pad_v'} if f != 1. else {'v_edit'}
    out = c.attend(q, k, v_edit, img, txt, act)
    assert c._v_txt is None
    full = fake_ctrl(f, 0., n_txt=n_txt).attend(q, k, v_edit, img, txt, act - {'v_edit'} | ({'pad_v'} if f != 1. else set()))
    assert torch.allclose(out[:, :, img], full[:, :, img], atol=1e-6)
    assert torch.allclose(out[:, :, txt], F.scaled_dot_product_attention(q[:, :, txt], k, v_orig), atol=1e-6)
    with pytest.raises(RuntimeError):                       # a kept V that attend() never read
        c.image_value(5, v_img, b, {'v_edit'})
        c.image_value(5, v_img, b, {'v_edit'})

    class Sub(Controller):
        def attend(self, q, k, v, img, txt, act):
            return super().attend(q, k, v, img, txt, act)
    s = object.__new__(Sub)
    s.basis_path, s.lam, s.capture, s.v_rows = 'x', 1.7, None, 'image'
    with pytest.raises(NotImplementedError):
        s.image_value(5, v_img, b, {'v_edit'})


def test_v_rows_config():
    assert config.load('sd35', 'ours').method.v_edit.rows == 'all' and config.load('flux', 'v_edit').method.v_edit.rows == 'all'
    assert config.load('flux', 'ours', ['v_edit.rows=image']).method.v_edit.rows == 'image'
