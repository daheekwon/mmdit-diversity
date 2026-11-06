"""Baselines: DAVE (Kwon et al., ICML 2026) and CADS (Sadat et al., ICLR 2024), both per image.

DAVE: before `block`, the image stream's token mean (DC) is scaled by alpha: h <- h - (1 - alpha) mean_tokens(h),
  computed in float64 (as value_diversity/ and diversity_anal/flux_dave.py).
CADS: every step, the text condition is replaced by sqrt(g) y + s sqrt(1 - g) n, rescaled to y's mean and std,
  g = linear schedule of t = 1 - step/T between tau1 and tau2; pooled conditioning unchanged. Noise and statistics are
  per image (each image has its own generator, seed + 1000000), so a batch gives the same images as batch-1 runs.
"""
import inspect
import math
import torch


def dave_hook(block, ctrl, alpha, steps, img_start=0):
    """img_start: first image row of hidden_states (FLUX.2 single blocks: the text length; the text rows pass unchanged).
    A controller in its unedited twin pass (E06 probe, ctrl.phase == 'twin') is left unedited."""
    sig = inspect.signature(block.forward)

    def hook(module, args, kwargs):
        if ctrl.step not in steps or getattr(ctrl, 'phase', 'main') == 'twin':
            return None
        bound = sig.bind(*args, **kwargs)
        x = bound.arguments['hidden_states']                    # image stream (FLUX.2 single blocks: [text | image])
        if img_start:
            raw = x[:, img_start:].double()
            out = x.clone()
            out[:, img_start:] = (raw-(1-alpha)*raw.mean(1, keepdim=True)).to(x.dtype)
            bound.arguments['hidden_states'] = out
        else:
            raw = x.double()
            bound.arguments['hidden_states'] = (raw-(1-alpha)*raw.mean(1, keepdim=True)).to(x.dtype)
        ctrl.base.append(('dave', ctrl.step))
        return bound.args, bound.kwargs
    return block.register_forward_pre_hook(hook, with_kwargs=True)


def cads_schedule(t, tau1, tau2):
    if t <= tau1:
        return 1.
    if t >= tau2:
        return 0.
    return (tau2-t)/(tau2-tau1)


def add_noise(y, gamma, noise_scale, psi, rescale, generator):
    """diversity_anal/cads_core.add_noise (A1111 rescaling semantics), applied to one image's rows."""
    if gamma == 1:
        return y
    y_mean, y_std = y.mean(), y.std(unbiased=False).clamp_min(1e-6)
    noise = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator)
    y2 = math.sqrt(gamma)*y+noise_scale*math.sqrt(1-gamma)*noise
    if not rescale:
        return y2
    y_scaled = (y2-y2.mean())/y2.std(unbiased=False).clamp_min(1e-6)*y_std+y_mean
    if torch.isnan(y_scaled).any():
        return y2
    return psi*y_scaled+(1-psi)*y2


class CADS:
    def __init__(self, c, seeds, steps):
        self.c, self.seeds, self.steps = c, list(seeds), steps
        self.clean = self.gens = None

    def __call__(self, module, args, kwargs, step):
        bound = inspect.signature(module.forward).bind(*args, **kwargs)
        text = bound.arguments['encoder_hidden_states']
        if step == 0:
            self.clean = text.detach().clone()
            self.gens = [torch.Generator(text.device).manual_seed(int(s)+1000000) for s in self.seeds]
        gamma = cads_schedule(1-step/self.steps, self.c.tau1, self.c.tau2)
        if gamma == 1:
            return None
        n = len(self.seeds)
        out = self.clean.clone()
        for j in range(n):                                      # SD3.5 rows [uncond | cond]: uncond first, as before
            for r in range(j, self.clean.shape[0], n):
                out[r:r+1] = add_noise(self.clean[r:r+1], gamma, self.c.noise_scale, self.c.psi, self.c.rescale, self.gens[j])
        bound.arguments['encoder_hidden_states'] = out
        return bound.args, bound.kwargs
