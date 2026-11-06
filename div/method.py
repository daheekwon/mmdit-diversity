"""The method and the baselines, driven by one Controller per pipeline call.

Components (each with its own blocks / steps / strength, see configs/<model>.yaml):
  v_edit  image V rows: V' = V + (lam-1) Q_b Q_b^T (V - b), Q_b = seed-contrast basis of block b (basis.py).
          Read by image and text queries alike (rows=all, default); rows=image: only image-query rows read V', text-query
          rows read the unedited image V (as pad_v; DESIGN_vrows_ko.md in experiments/E06_boundary). Text V, Q, K untouched.
          time_mean (video models only): the added value's mean over the latent frames at each position is added to
          every frame instead (E02 PROTOCOL_ko.md section 7c); with one frame this is the edit above.
  pad_v   image-query rows read the padding tokens' values x factor; the softmax is unchanged, text-query rows untouched.
  head    (undecided) image-query rows: the write from image keys of head h x (1 - beta (1 - sbar_h)),
          sbar_h = mean over image keys j of 1 - m_j^2 / e_j, m_j / e_j = mean over image queries of A_qj / A_qj^2.
  dave    DC (token mean) of the image stream entering one block x alpha (baselines.py).
  cads    noise on the text condition (baselines.py).
Every component works per image: no batch statistics and no reference run, so images do not depend on the batch.
Model specifics (token layout, where q / k / v are read) live in models.py.
FLUX.2-klein runs CFG as two transformer calls per step (conditional, then empty prompt): `step` counts denoising steps,
and every component acts on both calls of a step, as on both CFG rows of SD3.5. Its padding positions come from the
conditional prompt (all padding in the empty-prompt call too), as in SD3.5.
"""
from contextlib import contextmanager
import math
import torch
import torch.nn.functional as F
from . import baselines, models
from .config import parse_range, resolve

HEAD_CHUNK = 128
_BASES, _Q = {}, {}


def load_basis(path):
    path = str(resolve(path))
    if path not in _BASES:
        _BASES[path] = torch.load(path, weights_only=True)
    return _BASES[path]


def basis_unit(basis, blocks):
    """Relative image-V change per unit (lam - 1), averaged over the edited blocks."""
    u = [basis['unit'][b] for b in blocks]
    return u[0] if max(u) == min(u) else sum(u)/len(u)


class Controller:
    time_mean, frame_tokens = False, None
    v_rows, _v_txt = 'all', None                   # v_edit.rows; unedited image V kept for the text-query rows (rows=image)

    def __init__(self, pipe, cfg, prompt, seeds, capture=None):
        self.model, m, self.steps = cfg.model.name, cfg.method, cfg.model.steps
        nb = models.N_BLOCKS[self.model]
        self.cfg, self.seeds, self.capture = cfg, list(seeds), capture
        self.on = {c: bool(m[c].enabled) for c in ('v_edit', 'pad_v', 'head', 'dave', 'cads')}
        self.scope = {c: (set(parse_range(m[c].blocks, nb)), set(parse_range(m[c].steps, self.steps)))
                      for c in ('v_edit', 'pad_v', 'head') if self.on[c]}
        if capture is not None:
            self.scope['capture'] = (set(capture.blocks), set(capture.steps))
        self.lam = self.basis_path = None
        if self.on['v_edit']:
            self.basis_path = str(resolve(m.v_edit.basis))
            basis = load_basis(self.basis_path)
            if basis['model'] != self.model:
                raise ValueError(f"basis {m.v_edit.basis} is for {basis['model']}")
            blocks = sorted(self.scope['v_edit'][0])
            missing = [b for b in blocks if b not in basis['Q']]
            if missing:
                raise ValueError(f'basis has no Q for blocks {missing}')
            self.lam = float(m.v_edit.lam) if m.v_edit.lam is not None else 1+float(m.v_edit.strength)/basis_unit(basis, blocks)
            self.time_mean = bool(m.v_edit.get('time_mean', False))
            self.v_rows = str(m.v_edit.get('rows', 'all'))
            if self.v_rows not in ('all', 'image'):
                raise ValueError(f'v_edit.rows {self.v_rows!r}: all | image')
            if self.v_rows == 'image' and self.time_mean:
                raise NotImplementedError('v_edit.rows=image with time_mean')
            if self.time_mean:
                if self.model not in models.VIDEO:
                    raise ValueError('v_edit.time_mean is for video models')
                self.frame_tokens = models.frame_tokens(cfg.model)
        if self.on['cads'] and self.model == 'klein':
            raise NotImplementedError('CADS on FLUX.2-klein (two CFG calls per step) is not implemented')
        self.cfg_calls = models.CFG_CALLS.get(self.model, 1)
        pad_kw = dict(max_len=int(cfg.model.max_sequence_length)) if self.model == 'klein' else {}
        self.pad_index = models.pad_index(pipe, self.model, prompt, **pad_kw) if self.on['pad_v'] else None
        self.step, self.n_calls, self.calls, self.base = -1, 0, [], []

    # ------------------------------------------------------------------ scope
    def active(self, block):
        """Components acting on this block at the current step; logs the call when any does."""
        act = {c for c, (bs, ss) in self.scope.items() if block in bs and self.step in ss}
        if act:
            self.calls.append((self.step, block, tuple(sorted(act))))
        return act

    def expected_calls(self):
        nb = models.N_BLOCKS[self.model]
        return self.cfg_calls*sum(1 for s in range(self.steps) for b in range(nb)
                                  if any(b in bs and s in ss for bs, ss in self.scope.values()))

    def _q(self, block, device):
        key = (self.basis_path, block, device)
        if key not in _Q:
            _Q[key] = load_basis(self.basis_path)['Q'][block].to(device, torch.float32)
        return _Q[key]

    # ------------------------------------------------------------------ edits
    @torch.no_grad()
    def image_value(self, block, v, bias, act):
        """v: image V rows [B, n, C] (to_v output, bias included). Capture, then V edit.
        The edit is computed in float32, as in value_diversity/ (the parity tests compare pixels with its images). The
        float32 temporaries are reused in place: the result is bit-identical to (v.float() + (lam-1) d).to(v.dtype), and
        the extra peak memory is ~1.0 GiB instead of 2.3 GiB (1024, batch 10; tests/test_method.py checks the identity)."""
        if 'capture' not in act and 'v_edit' not in act:
            return v
        x = v.to(torch.float32, copy=True)              # a copy even when v is float32: x is changed in place
        if bias is not None:                            # bias None: FLUX.2 (no V bias)
            x.sub_(bias.float())
        if 'capture' in act:
            self.capture.add(block, x)
        if 'v_edit' not in act:
            return v
        if self.v_rows == 'image':                   # attend() reads this for the text-query rows
            if type(self).attend is not Controller.attend:
                raise NotImplementedError('v_edit.rows=image needs Controller.attend (a subclass replaced it)')
            if self._v_txt is not None:
                raise RuntimeError('unedited image V of the previous call was not read by attend()')
            self._v_txt = v
        Q = self._q(block, v.device)
        d = (x@Q)@Q.T
        del x
        if self.time_mean:
            d = frame_mean(d, self.frame_tokens)        # a new tensor (never a view of v)
        return d.mul_(self.lam-1).add_(v.float()).to(v.dtype)

    @torch.no_grad()
    def wants_attend(self, act):
        """Whether a model adapter must route this call's attention through attend(): pad_v or head acts here, or a
        subclass replaced attend (e.g. to measure attention). SD3.5 always calls attend; FLUX otherwise keeps its own
        attention dispatch, which the parity tests check pixel for pixel."""
        return ('pad_v' in act or 'head' in act or (self.v_rows == 'image' and 'v_edit' in act)
                or type(self).attend is not Controller.attend)

    def attend(self, q, k, v, img, txt, act):
        """Joint attention [B, H, S, D] with pad_v / head on the image-query rows; img / txt: slices of the sequence.
        v_edit.rows=image: the text-query rows read the unedited image V kept by image_value()."""
        vt = v
        if self._v_txt is not None:
            x, self._v_txt = self._v_txt, None
            B, H, _, D = v.shape
            vt = v.clone()
            vt[:, :, img] = x.reshape(B, -1, H, D).transpose(1, 2).to(v.dtype)
        if 'pad_v' not in act and 'head' not in act and vt is v:
            return F.scaled_dot_product_attention(q, k, v)
        vi = v
        if 'pad_v' in act:
            pos = (txt.start+self.pad_index).to(v.device)
            vi = v.clone()
            vi[:, :, pos] = (v[:, :, pos].float()*self.cfg.method.pad_v.factor).to(v.dtype)
        out_img = F.scaled_dot_product_attention(q[:, :, img], k, vi)
        if 'head' in act:
            out_img = (out_img.float()+head_delta(q[:, :, img], k, v, img, self.cfg.method.head.beta)).to(out_img.dtype)
        out_txt = F.scaled_dot_product_attention(q[:, :, txt], k, vt)
        return torch.cat([out_img, out_txt] if img.start == 0 else [out_txt, out_img], 2)

    # ------------------------------------------------------------------ install
    @contextmanager
    def install(self, pipe):
        """Set the attention processors and the step / baseline hooks for one pipeline call."""
        tr = pipe.transformer
        self.step, self.n_calls, self.calls = -1, 0, []
        handles = [tr.register_forward_pre_hook(self._start, with_kwargs=True)]
        if self.on['dave']:
            d = self.cfg.method.dave
            img_start = 0                      # FLUX.2 single blocks take one [text | image] stream: DC of the image rows
            if self.model == 'klein' and d.block >= models.KLEIN_DOUBLE:
                img_start = models.n_text_tokens(self.cfg.model)
            handles.append(baselines.dave_hook(models.blocks(pipe, self.model)[d.block], self, d.alpha,
                                               set(parse_range(d.steps, self.steps)), img_start=img_start))
        self.cads = baselines.CADS(self.cfg.method.cads, self.seeds, self.steps) if self.on['cads'] else None
        restore = models.set_processors(pipe, self.model, self)
        try:
            yield self
        finally:
            restore()
            for h in handles:
                h.remove()

    def _start(self, module, args, kwargs):
        self.n_calls += 1
        if (self.n_calls-1) % self.cfg_calls:          # second CFG call of the same step (FLUX.2-klein)
            return None
        self.step += 1
        if self.cads is not None:
            return self.cads(module, args, kwargs, self.step)

    def verify(self):
        if self.step != self.steps-1 or self.n_calls != self.steps*self.cfg_calls:
            raise RuntimeError(f'saw {self.n_calls} transformer calls ({self.step+1} steps), expected {self.steps} steps '
                               f'x {self.cfg_calls}')
        if len(self.calls) != self.expected_calls():
            raise RuntimeError(f'edited {len(self.calls)} attention calls, expected {self.expected_calls()}')
        if self.on['dave'] and len(self.base) != self.cfg_calls*len(parse_range(self.cfg.method.dave.steps, self.steps)):
            raise RuntimeError('DAVE did not run on every configured step')

    def info(self):
        return dict(lam=self.lam, time_mean=self.time_mean, basis=self.basis_path,
                    pad_tokens=None if self.pad_index is None else len(self.pad_index), attention_calls=len(self.calls))


def frame_mean(d, per_frame):
    """d: [B, frames x per_frame, C], frame-major -> the mean over frames at each position, repeated on every frame."""
    B, n, C = d.shape
    if n % per_frame:
        raise ValueError(f'{n} video rows are not whole frames of {per_frame}')
    f = n//per_frame
    return d.view(B, f, per_frame, C).mean(1, keepdim=True).expand(B, f, per_frame, C).reshape(B, n, C)


def head_delta(qi, k, v, img, beta):
    """-beta (1 - sbar_h) A_II V_I for the image-query rows qi [B, H, n, D] (float32)."""
    n, scale = qi.shape[2], 1/math.sqrt(qi.shape[-1])
    kf, vi = k.float(), v[:, :, img].float()
    colsum = torch.zeros(*qi.shape[:2], n, dtype=torch.float64, device=qi.device)
    colsq, uni = torch.zeros_like(colsum), []
    for c in range(0, n, HEAD_CHUNK):
        a = (qi[:, :, c:c+HEAD_CHUNK].float()@kf.transpose(-1, -2)*scale).softmax(-1)[..., img]
        uni.append(a@vi)
        colsum += a.sum(2, dtype=torch.float64)
        colsq += a.square().sum(2, dtype=torch.float64)
    m, e = colsum/n, colsq/n
    sbar = (1-m.square()/e.clamp_min(1e-300)).clamp_min(0.).mean(-1).float()
    return -beta*(1-sbar)[..., None, None]*torch.cat(uni, 2)
