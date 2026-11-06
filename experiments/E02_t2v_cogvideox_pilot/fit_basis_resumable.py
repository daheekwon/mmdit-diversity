"""scripts/fit_basis.py with a checkpoint after every calibration prompt (shared GPUs: an out-of-memory error ends the
process and gpu_job.sh restarts it). Same capture as div/basis.capture (div/basis.Moments, prompt i in half i % 2,
prompts in file order), same fit and output files as scripts/fit_basis.py (bases/<name>.pt, .json, _moments.pt).

  python experiments/E02_t2v_cogvideox_pilot/fit_basis_resumable.py --name cogvideox_pilot_k256 \
      --prompts /DATA0/dahee/data/datasets/vbench/calib_dimension_longer_32.txt --seeds 8 --gpu 0

The checkpoint bases/<name>_capture_ckpt.pt holds the raw moment sums (float64) and the number of prompts done; it is
removed after the fit.
"""
import argparse
from pathlib import Path
import sys
import time
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
from div import basis, config, data, models                                   # noqa: E402
from div.method import Controller                                             # noqa: E402
from fit_basis import BASES, save                                             # noqa: E402


def capture_resumable(pipe, cfg, prompts, seeds, blocks, steps, ckpt, stop_after=None, mom=None):
    """basis.capture, saving the moment sums to ckpt after every prompt and starting from ckpt when it exists.
    stop_after (tests only): return None after this many prompts in this call.
    mom: a basis.Moments (or subclass, e.g. fit_basis_frame_mean.FrameMeanMoments) made for these blocks, steps, seeds."""
    mom = basis.Moments(blocks, steps, len(seeds)) if mom is None else mom
    start, key = 0, dict(prompts=list(prompts), seeds=list(seeds), blocks=list(blocks), steps=list(steps),
                         moments=type(mom).__name__)
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        if c['key'] != key:
            raise SystemExit(f'{ckpt} was made with other prompts, seeds, blocks or steps')
        mom.acc, start = c['acc'], c['done']
        print(f'resumed after {start} prompts', flush=True)
    for i in range(start, len(prompts)):
        t0 = time.monotonic()
        mom.half = i % 2
        ctrl = Controller(pipe, cfg, prompts[i], seeds, capture=mom)
        models.generate(pipe, cfg.model, models.encode(pipe, cfg.model, prompts[i]), seeds, ctrl, last_step=max(steps))
        tmp = ckpt.with_name(ckpt.name+'.tmp')
        torch.save(dict(key=key, acc=mom.acc, done=i+1), tmp)
        tmp.rename(ckpt)
        print(f'captured {i+1}/{len(prompts)} {prompts[i][:50]!r} {time.monotonic()-t0:.0f}s', flush=True)
        if stop_after is not None and i+1-start >= stop_after and i+1 < len(prompts):
            return None
    return mom.state()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--name', required=True)
    a.add_argument('--prompts', required=True)
    a.add_argument('--seeds', type=int, required=True)
    a.add_argument('--k', type=int, default=256)
    a.add_argument('--beta', type=float, default=1.)
    a.add_argument('--gpu', type=int, default=0)
    x = a.parse_args()
    cfg = config.load('cogvideox', 'original')
    cfg.data.calibration.prompts = x.prompts
    prompts = data.calibration_prompts(cfg.data)
    seeds = [cfg.data.calibration.seed_base+i for i in range(x.seeds)]
    blocks = config.parse_range('all', models.N_BLOCKS['cogvideox'])
    steps = config.parse_range(cfg.method.v_edit.steps, cfg.model.steps)
    pipe = models.load(cfg.model, x.gpu)
    ckpt = BASES/f'{x.name}_capture_ckpt.pt'
    state = capture_resumable(pipe, cfg, prompts, seeds, blocks, steps, ckpt)
    bias = {b: models.attentions(pipe, 'cogvideox')[b].to_v.bias.detach().float().cpu() for b in blocks}
    mom = dict(state=state, bias=bias, prompts=prompts, seeds=seeds, steps=steps)
    torch.save(mom, BASES/f'{x.name}_moments.pt')
    save(basis.fit(state, bias, 'cogvideox', x.k, x.beta, dict(prompts=prompts, seeds=seeds, steps=steps)), x.name)
    ckpt.unlink()


if __name__ == '__main__':
    main()
