"""V-edit basis: per block, the top-k eigenvectors of D/tr D - beta C/tr C (contrastive PCA) of the bias-free image V.

Capture (Original model, no edit): each calibration prompt runs once as a batch of N seeds up to the last capture step.
For every (prompt, step, CFG branch) group and block, with x the bias-free image V rows of the N seeds and m their
seed mean (per token):
  D += sum over seeds and tokens of (x - m)^T (x - m)    seed deviation
  C += sum over tokens of m^T m                          seed-common
  H += sum over tokens of m
The products of one group are computed in float32 (TF32 is off, models.load) and accumulated in float64 on the CPU.
Against float64 products (2026-11-04, CogVideoX real V, 30 blocks): moments, unit and selectivity agree within 1.1e-6;
the top-256 subspaces differ only where the contrast eigenvalues at the cut are ~0 (blocks 0, 1, 4), and there the
edit difference carries <= 9e-7 of the V energy. This step is 36x faster (video: 17,550 tokens per seed).
Prompts alternate between two halves (A, B). Fit pools both halves; the held-out check fits on one half and measures
selectivity = (share of the other half's D in Q) / (share of its C in Q), both directions.
unit_b = sqrt(tr(Q^T M Q) / E||V||^2), M = D + C, E||V||^2 = tr M + 2 b.H + b.b: the relative image-V change per unit
(lam - 1). It turns a strength (relative V change) into lam (method.py).
"""
import json
import time
import torch
from . import models
from .config import ROOT

LEGACY = ROOT.parent/'value_diversity'/'results'


class Moments:
    def __init__(self, blocks, steps, n_seeds):
        self.blocks, self.steps, self.n = list(blocks), list(steps), n_seeds
        self.half, self.acc = 0, {}

    @torch.no_grad()
    def add(self, block, x):
        """x: bias-free image V rows [B, n_tokens, C] of one block at one step; B = groups x N seeds."""
        if x.shape[0] % self.n:
            raise ValueError('batch is not a multiple of the seed count')
        a = self.acc.setdefault((self.half, block), dict(D=0., C=0., H=0., nD=0, nC=0))
        for g in range(x.shape[0]//self.n):
            X = x[g*self.n:(g+1)*self.n].float()
            m = X.mean(0)
            d = (X-m).flatten(0, 1)
            a['D'] = a['D']+(d.T@d).cpu().double()
            a['C'] = a['C']+(m.T@m).cpu().double()
            a['H'] = a['H']+m.sum(0).cpu().double()
            a['nD'] += d.shape[0]
            a['nC'] += m.shape[0]

    def state(self):
        return {f'{h}_{b}': dict(D=(a['D']/a['nD']).float(), C=(a['C']/a['nC']).float(), H=(a['H']/a['nC']).float(),
                                 nD=a['nD'], nC=a['nC']) for (h, b), a in self.acc.items()}


class StepSets:
    """Several Moments filled in one capture run, one per step set (e.g. {'s0-4': [0..4], 's0-6': [0..6]}): a call at
    step s goes to every set that holds s. Each set's state equals a separate capture over that set's steps. The step
    is read from the Controller that run_prompt binds (ctrl.step), so Controller.image_value stays unchanged."""

    def __init__(self, blocks, step_sets, n_seeds):
        self.sets = {name: Moments(blocks, steps, n_seeds) for name, steps in step_sets.items()}
        self.blocks = list(blocks)
        self.steps = sorted({s for steps in step_sets.values() for s in steps})
        self.ctrl, self._half = None, 0

    @property
    def half(self):
        return self._half

    @half.setter
    def half(self, h):
        self._half = h
        for m in self.sets.values():
            m.half = h

    def add(self, block, x):
        for m in self.sets.values():
            if self.ctrl.step in m.steps:
                m.add(block, x)

    def state(self):
        return {name: m.state() for name, m in self.sets.items()}


def capture_sets(pipe, cfg, prompts, seeds, blocks, step_sets):
    """capture() for several step sets at once (StepSets); returns {set name: moment state}."""
    from .method import Controller
    mom = StepSets(blocks, step_sets, len(seeds))
    for i, prompt in enumerate(prompts):
        t0 = time.monotonic()
        mom.half = i % 2
        ctrl = Controller(pipe, cfg, prompt, seeds, capture=mom)
        mom.ctrl = ctrl
        models.generate(pipe, cfg.model, models.encode(pipe, cfg.model, prompt), seeds, ctrl, last_step=max(mom.steps))
        print(f'captured {i+1}/{len(prompts)} {prompt[:50]!r} {time.monotonic()-t0:.0f}s', flush=True)
    return mom.state()


def capture(pipe, cfg, prompts, seeds, blocks, steps):
    """Run the calibration prompts and return the moment state (see Moments)."""
    from .method import Controller
    mom = Moments(blocks, steps, len(seeds))
    for i, prompt in enumerate(prompts):
        t0 = time.monotonic()
        mom.half = i % 2
        ctrl = Controller(pipe, cfg, prompt, seeds, capture=mom)
        models.generate(pipe, cfg.model, models.encode(pipe, cfg.model, prompt), seeds, ctrl, last_step=max(steps))
        print(f'captured {i+1}/{len(prompts)} {prompt[:50]!r} {time.monotonic()-t0:.0f}s', flush=True)
    return mom.state()


def _pool(state, block, halves=(0, 1)):
    parts = [state[f'{h}_{block}'] for h in halves if f'{h}_{block}' in state]
    nD, nC = sum(p['nD'] for p in parts), sum(p['nC'] for p in parts)
    return dict(D=sum(p['D'].double()*p['nD'] for p in parts)/nD, C=sum(p['C'].double()*p['nC'] for p in parts)/nC,
                H=sum(p['H'].double()*p['nC'] for p in parts)/nC)


def _contrast(s, k, beta):
    return torch.linalg.eigh(s['D']/torch.trace(s['D'])-beta*s['C']/torch.trace(s['C']))[1][:, -k:]


def _share(Q, S):
    return float(torch.trace(Q.T@S@Q)/torch.trace(S))


def fit(state, bias, model, k=256, beta=1., meta=None):
    """state: Moments.state(); bias: {block: to_v bias}. Returns the basis dict saved by scripts/fit_basis.py."""
    blocks = sorted({int(key.split('_')[1]) for key in state})
    Q, unit, check = {}, {}, {}
    for b in blocks:
        s = _pool(state, b)
        q = _contrast(s, k, beta)
        M, bb = s['D']+s['C'], bias[b].double()
        Q[b] = q.float()
        unit[b] = float((torch.trace(q.T@M@q)/(torch.trace(M)+2*bb@s['H']+bb@bb)).sqrt())
        if f'0_{b}' in state and f'1_{b}' in state:
            sel = []
            for fit_h, test_h in ((0, 1), (1, 0)):
                qh, t = _contrast(_pool(state, b, (fit_h,)), k, beta), _pool(state, b, (test_h,))
                sel.append(_share(qh, t['D'])/_share(qh, t['C']))
            check[b] = dict(selectivity_ab=sel[0], selectivity_ba=sel[1], energy_share=_share(q, M))
    return dict(model=model, k=k, beta=beta, blocks=blocks, Q=Q, unit=unit, check=check, source='fit', **(meta or {}))


def legacy(model, all_blocks=False):
    """The dev-8-prompt bases of value_diversity/ (all earlier results). Their units are the pooled legacy values (one
    per fitted block range), so lam matches the old runs exactly. SD3.5 all_blocks: the merged 0-23 file of the
    all-block runs (value_diversity/results/generalize/allblock, combo_sd35 'qa'); its blocks 6-23 equal the 6-23 basis."""
    if model == 'sd35':
        root = LEGACY/'sd35_v1'/'subspace_edit'/'phase0'
        unit_of = lambda r: json.loads((r/'FIT_REPORT.json').read_text())['final']['256']['relative_change_per_unit_lambda']['contrast']
        Q = torch.load(root/('early_blocks/subspaces_k256_b0to23.pt' if all_blocks else 'subspaces_k256.pt'),
                       weights_only=True)['contrast']
        u_mid, u_early = unit_of(root), unit_of(root/'early_blocks') if all_blocks else None
        unit = {b: u_early if int(b) < 6 else u_mid for b in Q}
    elif all_blocks:
        raise ValueError('no all-block legacy basis for flux')
    else:
        root = LEGACY/'flux_qedit_v1'/'phase0'
        Q = torch.load(root/'subspaces_k256.pt', weights_only=True)['contrast']
        rep = json.loads((root/'FIT_REPORT.json').read_text())
        u = rep['final']['256']['relative_change_per_unit_lambda']
        unit = {b: u[name]['contrast'] for name, bs in rep['ranges'].items() for b in bs}
    Q = {int(b): q.float() for b, q in Q.items()}
    return dict(model=model, k=256, beta=1., blocks=sorted(Q), Q=Q, unit={int(b): unit[b] for b in Q}, check={},
                source=f'legacy: {root} (8 dev prompts x 16 calibration seeds; parity tests only)')
