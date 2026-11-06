"""E06 stage-2 probes (PROTOCOL_ko.md §9). One Controller subclass, so the method's own edits (V edit, pad V) run as in
stage 1; the probe adds, per option:

  signal_steps  record internal signals at these denoising steps (main pass, all blocks):
                I->T concept-column probabilities of the image queries (head mean, 2x2-pooled patches),
                concept-token keys (random 128-d projection of the head-concatenated key),
                block-input LayerNorm sigma and L2 norm of the image tokens,
                V-edit-basis coordinates c = Q^T (V - b) of the image tokens (token mean and RMS, pre-edit),
                and x0-hat = x_t - sigma * v (guided) at X0_STEPS.  Conditional rows only.
  twin          E3: at every edit step, run the transformer once more on the same x_t with no edit (twin pass) and
                fix, in the main pass, 'it' = the image-query probabilities on the concept columns (other columns
                rescaled so each row sums to 1; conditional rows only) and/or 'txt' = the text-token states at every
                block input (both CFG rows).
                Read-level text fixes (DESIGN_e3_split_ko.md A; one run uses these alone): in every block the image
                queries read the twin's text keys and / or values: 'tk' / 'tv' = all text columns, 'wk' / 'wv' = the
                prompt-word columns only (both CFG rows). Added as SDPA(q_img, K', V') - SDPA(q_img, K, V) in float32,
                so twin = main gives exactly zero. 'tk' + 'tv' is the text-stream fix written at the image-query read.
  text_sig      with signal_steps: also record text V of the prompt words and K / V of the other text tokens (pad and
                special tokens), each a fixed random 128-d projection (DESIGN_e3_split_ko.md B0).
  write_basis   open-loop write measure (no edit applied): at the V-edit blocks / steps, the change of the block's
                residual write gate * W_out (A dV) for dV = Q Q^T x and R R^T x (x = V - b), A = this call's attention.

The twin pass runs inside the transformer pre-hook, so the step counter and the edit log see one call per step.
"""
import inspect
import math
import torch
import torch.nn.functional as F
from div import models
from div.config import parse_range
from div.method import Controller, load_basis

CHUNK = 256
KPROJ = 128
READ = {'tk', 'tv', 'wk', 'wv'}           # read-level text fixes (image queries read the twin's text K / V)
X0_STEPS = (0, 1, 2, 3, 5, 8, 11, 12, 15, 20, 27)


def concept_index(pipe, model, prompt, max_len=256):
    """Prompt-word positions in the text part of the joint sequence (no BOS / EOS / padding)."""
    if model == 'sd35':
        tok, t5 = pipe.tokenizer, pipe.tokenizer_3
        ci = tok(prompt, padding='max_length', max_length=models.CLIP_LEN, truncation=True)['input_ids']
        first_eos = ci.index(tok.eos_token_id)
        ti = t5(prompt, padding='max_length', max_length=max_len, truncation=True, add_special_tokens=True)['input_ids']
        pos = list(range(1, first_eos))+[models.CLIP_LEN+i for i, t in enumerate(ti)
                                          if t not in (t5.pad_token_id, t5.eos_token_id)]
    elif model == 'flux':
        t5 = pipe.tokenizer_2
        ids = t5(prompt, padding='max_length', max_length=max_len, truncation=True)['input_ids']
        pos = [i for i, t in enumerate(ids) if t not in (t5.pad_token_id, t5.eos_token_id)]
    else:
        raise ValueError(model)
    return torch.tensor(pos, dtype=torch.long)


class _Cap:
    """Controller capture hook: keeps the last image-V rows (V - b) per block for the probe."""

    def __init__(self, blocks, steps):
        self.blocks, self.steps, self.x = list(blocks), list(steps), {}

    def add(self, block, x):
        self.x[block] = x


class Probe(Controller):
    def __init__(self, pipe, cfg, prompt, seeds, signal_steps=(), twin=(), write_basis=None, text_sig=False):
        model = cfg.model.name
        nb = models.N_BLOCKS[model]
        steps = cfg.model.steps
        m = cfg.method
        self.vblocks = sorted(parse_range(m.v_edit.blocks, nb))
        self.vsteps = set(parse_range(m.v_edit.steps, steps))
        self.signal_steps, self.twin_fix = set(signal_steps), set(twin)
        if self.twin_fix - {'it', 'txt'} - READ:
            raise ValueError(twin)
        if self.twin_fix & READ and self.twin_fix & {'it', 'txt'}:
            raise ValueError(f'read-level text fixes run alone: {twin}')
        if {'tk', 'wk'} <= self.twin_fix or {'tv', 'wv'} <= self.twin_fix:
            raise ValueError(f'one column set per K / V: {twin}')
        self.text_sig = bool(text_sig)
        if self.text_sig and not self.signal_steps:
            raise ValueError('text_sig needs signal_steps')
        self.write = write_basis is not None
        if self.write and (m.v_edit.enabled or m.pad_v.enabled):
            raise ValueError('the write measure runs on the unedited model')
        cap_steps = self.signal_steps | (self.vsteps if self.write else set())
        cap = _Cap(self.vblocks, sorted(cap_steps)) if cap_steps else None
        super().__init__(pipe, cfg, prompt, seeds, capture=cap)
        edit_steps = set()
        for c in ('v_edit', 'pad_v'):
            if self.on[c]:
                edit_steps |= self.scope[c][1]
        if self.on['dave']:                    # §17: DAVE with the same blocks; its hook skips the twin pass
            edit_steps |= set(parse_range(m.dave.steps, steps))
        self.twin_steps = edit_steps if self.twin_fix else set()
        if self.twin_fix and not edit_steps:
            raise ValueError('twin needs an edit')
        probe_steps = self.signal_steps | self.twin_steps
        if probe_steps:
            self.scope['probe'] = (set(range(nb)), probe_steps)
        self.pipe, self.nb = pipe, nb
        self.concept = concept_index(pipe, model, prompt)
        self.basis_q = load_basis(m.v_edit.basis)['Q'] if self.signal_steps or self.write else None
        self.basis_r = load_basis(write_basis)['Q'] if self.write else None
        g = torch.Generator().manual_seed(0)
        self.kproj = None
        self.vproj, self._gv = None, torch.Generator().manual_seed(1)   # own generator: the K projection is unchanged
        self.phase, self.cur = 'main', None
        self.twin_it, self.twin_txt, self.twin_kv, self.gate = {}, {}, {}, {}
        self.sig = dict(it={}, kc={}, ln={}, c={}, x0={})
        if self.text_sig:
            self.sig.update(vc={}, ko={}, vo={})
        self.energy = {}                       # write measure: block -> [eQ, eR, e_native]
        self._g = g
        self._in = None

    # ------------------------------------------------------------------ bookkeeping
    def rows(self, B):
        """Conditional rows of a batch (SD3.5: [uncond | cond])."""
        return slice(B//2, B) if self.model == 'sd35' else slice(0, B)

    def active(self, block):
        self.cur = block
        if self.phase == 'twin':
            return {'twin'}
        return super().active(block)

    def _start(self, module, args, kwargs):
        r = super()._start(module, args, kwargs)
        self._in = kwargs
        if self.step in self.twin_steps:
            self.phase, self.twin_it, self.twin_txt, self.twin_kv = 'twin', {}, {}, {}
            try:
                module.forward(*args, **kwargs)
            finally:
                self.phase = 'main'
        return r

    def install(self, pipe):
        cm = super().install(pipe)
        probe = self

        class _Both:
            def __enter__(self_):
                cm.__enter__()
                self_.h = []
                tr = pipe.transformer
                for b, blk in enumerate(models.blocks(pipe, probe.model)):
                    self_.h.append(blk.register_forward_pre_hook(probe._block_pre(b), with_kwargs=True))
                    if probe.write:
                        norm = blk.norm if hasattr(blk, 'norm') and not hasattr(blk, 'norm1') else blk.norm1
                        self_.h.append(norm.register_forward_hook(probe._gate_hook(b)))
                if probe.signal_steps:
                    self_.h.append(tr.register_forward_hook(probe._x0_hook, with_kwargs=True))
                return probe

            def __exit__(self_, *exc):
                for h in self_.h:
                    h.remove()
                return cm.__exit__(*exc)
        return _Both()

    # ------------------------------------------------------------------ hooks
    @staticmethod
    def _as_kwargs(module, args, kwargs):
        """The block call with every argument by name (an earlier pre-hook, DAVE, re-binds the call positionally)."""
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        out = {}
        for name, p in bound.signature.parameters.items():
            if name not in bound.arguments:
                continue
            if p.kind is p.VAR_KEYWORD:
                out.update(bound.arguments[name])
            elif p.kind is p.VAR_POSITIONAL:
                raise TypeError('block forward with *args')
            else:
                out[name] = bound.arguments[name]
        return out

    def _block_pre(self, b):
        def hook(module, args, kwargs):
            rebound = 'encoder_hidden_states' not in kwargs
            if rebound:
                args, kwargs = (), self._as_kwargs(module, args, kwargs)
            if self.phase == 'twin':
                if 'txt' in self.twin_fix:
                    self.twin_txt[b] = kwargs['encoder_hidden_states'].detach().clone()
                return (args, kwargs) if rebound else None
            if self.step in self.signal_steps:             # image tokens: the same with or without the text fix
                h = kwargs['hidden_states']
                h = h[self.rows(h.shape[0])].float()
                sd = h.std(-1, unbiased=False).mean(-1)
                nm = h.norm(dim=-1).mean(-1)
                self.sig['ln'][(self.step, b)] = torch.stack([sd, nm], -1).cpu()
            if self.step in self.twin_steps and 'txt' in self.twin_fix:
                kwargs = dict(kwargs)
                kwargs['encoder_hidden_states'] = self.twin_txt[b]
                return args, kwargs
            return (args, kwargs) if rebound else None
        return hook

    def _gate_hook(self, b):
        def hook(module, args, out):
            self.gate[b] = out[1].detach()
        return hook

    def _x0_hook(self, module, args, kwargs, out):
        if self.phase != 'main' or self.step not in X0_STEPS:
            return
        x, t = kwargs['hidden_states'].float(), kwargs['timestep'].float()
        v = out[0].float()
        if self.model == 'sd35':
            B = x.shape[0]//2
            v = v[:B]+self.cfg.model.guidance*(v[B:]-v[:B])
            x, sigma = x[B:], t[B:]/1000
        else:
            sigma = t
        x0 = x-sigma.view(-1, *[1]*(x.ndim-1))*v
        self.sig['x0'][self.step] = x0.to(torch.float16).cpu()

    # ------------------------------------------------------------------ attention
    def _probs_concept(self, q, k, img, txt):
        """Probabilities of the image queries on the concept columns [B, H, n, nc] (float32)."""
        pos = (txt.start+self.concept).to(q.device)
        scale = 1/math.sqrt(q.shape[-1])
        kf = k.float().transpose(-1, -2)
        out = []
        qi = q[:, :, img]
        for c in range(0, qi.shape[2], CHUNK):           # the same ops as _fix_delta, so twin = main gives tp = P exactly
            out.append((qi[:, :, c:c+CHUNK].float()@kf*scale).softmax(-1)[..., pos])
        return torch.cat(out, 2)

    def _fix_delta(self, q, k, v, img, txt, tp):
        """(P_fixed - P) V for the image queries (float32): P_fixed has the concept-column probabilities tp and the
        other columns rescaled so each row sums to 1. Added to the native attention output, so that with tp = P (twin =
        main, the null check) the correction is exactly zero and the image equals the unfixed run."""
        pos = (txt.start+self.concept).to(q.device)
        scale = 1/math.sqrt(q.shape[-1])
        kf, vf = k.float().transpose(-1, -2), v.float()
        qi, out = q[:, :, img], []
        for c in range(0, qi.shape[2], CHUNK):
            p = (qi[:, :, c:c+CHUNK].float()@kf*scale).softmax(-1)
            t = tp[:, :, c:c+CHUNK]
            f = (1-t.sum(-1, keepdim=True))/(1-p[..., pos].sum(-1, keepdim=True)).clamp_min(1e-12)
            d = p*(f-1)
            d[..., pos] = t-p[..., pos]
            out.append(d@vf)
        return torch.cat(out, 2)

    def _read_delta(self, q, k, v, img, txt, act, tk, tv):
        """SDPA(q_img, K', V') - SDPA(q_img, K, V) for the image queries of every row (float32). K' / V': the twin's
        text keys / values (tk, tv: [B, H, n_txt, D]) on the fixed columns; pad V is shrunk after the swap, as the image
        queries read it. Both terms use the same ops, so twin = main gives exactly zero."""
        f = self.twin_fix
        ts, te, _ = txt.indices(k.shape[2])               # txt may end at None (SD3.5: [image | text])
        allc = torch.arange(te-ts, device=q.device)
        word = self.concept.to(q.device)
        kc = allc if 'tk' in f else word if 'wk' in f else None
        vc = allc if 'tv' in f else word if 'wv' in f else None
        fac = self.cfg.method.pad_v.factor if 'pad_v' in act else None
        ppos = (txt.start+self.pad_index).to(q.device) if fac is not None else None
        scale = 1/math.sqrt(q.shape[-1])
        kf, vb = k.float(), v.float()
        if fac is not None:
            vb = vb.clone()
            vb[:, :, ppos] = vb[:, :, ppos]*fac
        k2, v2 = kf, vb
        if kc is not None:
            k2 = kf.clone()
            k2[:, :, txt.start+kc] = tk[:, :, kc].float()
        if vc is not None:
            v2 = v.float().clone()
            v2[:, :, txt.start+vc] = tv[:, :, vc].float()
            if fac is not None:
                v2[:, :, ppos] = v2[:, :, ppos]*fac
        kt, k2t = kf.transpose(-1, -2), k2.transpose(-1, -2)
        cols = txt.start+(vc if vc is not None else allc)
        dv = (v2-vb)[:, :, cols]
        qi, out = q[:, :, img], []
        for c in range(0, qi.shape[2], CHUNK):
            qc = qi[:, :, c:c+CHUNK].float()
            p = (qc@kt*scale).softmax(-1)
            if kc is None:                                 # same keys: only the swapped values change the output
                out.append(p[..., cols]@dv)
            else:
                out.append((qc@k2t*scale).softmax(-1)@v2-p@vb)
        return torch.cat(out, 2)

    @torch.no_grad()
    def attend(self, q, k, v, img, txt, act):
        b = self.cur
        r = self.rows(q.shape[0])
        if self.phase == 'twin':
            if 'it' in self.twin_fix:
                self.twin_it[b] = self._probs_concept(q[r], k[r], img, txt)
            if self.twin_fix & READ:
                self.twin_kv[b] = (k[:, :, txt].clone(), v[:, :, txt].clone())
            return F.scaled_dot_product_attention(q, k, v)
        if 'head' in act:
            raise NotImplementedError('head with the probe')
        out = super().attend(q, k, v, img, txt, act)          # the unfixed run's own numerics
        if self.step in self.twin_steps and 'it' in self.twin_fix:
            vi = v
            if 'pad_v' in act:
                pos = (txt.start+self.pad_index).to(v.device)
                vi = v.clone()
                vi[:, :, pos] = (v[:, :, pos].float()*self.cfg.method.pad_v.factor).to(v.dtype)
            d = self._fix_delta(q[r], k[r], vi[r], img, txt, self.twin_it[b])
            out = out.clone()
            out[r, :, img] = (out[r, :, img].float()+d).to(out.dtype)
        if self.step in self.twin_steps and self.twin_fix & READ:
            d = self._read_delta(q, k, v, img, txt, act, *self.twin_kv[b])
            out = out.clone()
            out[:, :, img] = (out[:, :, img].float()+d).to(out.dtype)
        if self.step in self.signal_steps:
            self._record(b, q[r], k[r], img, txt, v[r])
        if self.write and 'capture' in act:
            self._write(b, q, k, img)
        return out

    def _proj(self, x, pos, which):
        """[B, H, S, D] rows at pos -> fixed random 128-d projection of the head-concatenated vectors [B, len, 128]."""
        xc = x[:, :, pos].float().permute(0, 2, 1, 3).reshape(x.shape[0], len(pos), -1)
        if which == 'k':
            if self.kproj is None or self.kproj.shape[0] != xc.shape[-1]:
                self.kproj = (torch.randn(xc.shape[-1], KPROJ, generator=self._g)/math.sqrt(KPROJ))
            P = self.kproj
        else:
            if self.vproj is None or self.vproj.shape[0] != xc.shape[-1]:
                self.vproj = (torch.randn(xc.shape[-1], KPROJ, generator=self._gv)/math.sqrt(KPROJ))
            P = self.vproj
        return (xc@P.to(xc.device)).to(torch.float16).cpu()

    def _record(self, b, q, k, img, txt, v=None):
        s = self.step
        p = self._probs_concept(q, k, img, txt).mean(1)                       # [B, n, nc]
        B, n, nc = p.shape
        side = int(math.isqrt(n))
        p = p.view(B, side, side, nc).permute(0, 3, 1, 2)
        p = F.avg_pool2d(p, 2).permute(0, 2, 3, 1).reshape(B, -1, nc)
        self.sig['it'][(s, b)] = p.to(torch.float16).cpu()
        pos = (txt.start+self.concept).to(k.device)
        self.sig['kc'][(s, b)] = self._proj(k, pos, 'k')                          # [B, nc, 128]
        if self.text_sig:
            word = set(self.concept.tolist())
            ts, te, _ = txt.indices(k.shape[2])
            other = torch.tensor([i for i in range(te-ts) if i not in word], device=k.device)+ts
            self.sig['vc'][(s, b)] = self._proj(v, pos, 'v')
            self.sig['ko'][(s, b)] = self._proj(k, other, 'k')
            self.sig['vo'][(s, b)] = self._proj(v, other, 'v')
        x = self.capture.x.pop(b, None) if self.capture is not None else None
        if x is not None and b in self.basis_q:
            Q = self.basis_q[b].to(x.device)
            c = x[self.rows(x.shape[0])].float()@Q                                # [B, n, k]
            self.sig['c'][(s, b)] = torch.stack([c.mean(1), c.square().mean(1).sqrt()], 1).cpu()

    def _write(self, b, q, k, img):
        x = self.capture.x.pop(b).float()                                         # [B, n, C]
        B, n, C = x.shape
        H, D = q.shape[1], q.shape[-1]
        blk = models.blocks(self.pipe, self.model)[b]
        W = blk.proj_out.weight[:, :C] if hasattr(blk, 'proj_out') and hasattr(blk, 'norm') and not hasattr(blk, 'norm1') \
            else blk.attn.to_out[0].weight
        W, gate = W.float(), self.gate[b].float()
        e = self.energy.setdefault(b, [0., 0., 0.])
        for j, basis in enumerate((self.basis_q, self.basis_r)):
            Bm = basis[b].to(x.device)
            dv = ((x@Bm)@Bm.T).view(B, n, H, D).transpose(1, 2)
            full = torch.zeros(B, H, q.shape[2], D, device=q.device, dtype=q.dtype)
            full[:, :, img] = dv.to(q.dtype)
            do = F.scaled_dot_product_attention(q[:, :, img], k, full).float()
            dw = (do.transpose(1, 2).reshape(B, n, H*D)@W.T)*gate[:, None, :]
            e[j] += float(dw.square().sum())
        e[2] += float(x.square().sum())

    def signals(self):
        """Per-image signal dicts (conditional rows in seed order), keys 'step,block' (x0: 'step')."""
        key = lambda k: ','.join(map(str, k)) if isinstance(k, tuple) else str(k)
        return [{kind: {key(k): val[i].clone() for k, val in d.items()} for kind, d in self.sig.items()}
                for i in range(len(self.seeds))]
