"""Stage-1 measurements of PROTOCOL_ko.md (Original model only; no method outcome is computed).

  python experiments/E02_t2v_cogvideox_pilot/measure.py cost --gpu 0
  python experiments/E02_t2v_cogvideox_pilot/measure.py capture --gpu 0
  python experiments/E02_t2v_cogvideox_pilot/measure.py padmass --gpu 0

cost     full setting (49 frames, 480x720, 50 steps), first VBench tune prompt, 2 seeds: wall time and peak GPU memory
         per call at batch 1 and batch 2; batch 2 vs batch 1 pixel comparison; the videos are kept for viewing.
capture  basis capture (div/basis.py) for one prompt up to the last V-edit step, N = 4 and 8 seeds in one call:
         wall time and peak memory (a capture call holds all N seeds of a prompt).
padmass  video-query softmax mass on the padding keys (mean over video queries and heads; share of video queries with
         mass > 0.5), every block, steps 0/5/10/15/20, cond and uncond rows; 8 tune prompts x 1 seed, long and short
         prompt versions. Each run stops after step 20.
Output: outputs/cogvideox/pilot/<command>.json
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from div import basis, config, data, models                                   # noqa: E402
from div.method import Controller                                             # noqa: E402

OUT = config.ROOT/'outputs'/'cogvideox'/'pilot'
CHUNK = 256
PAD_STEPS = (0, 5, 10, 15, 20)
SEEDS = [5100000, 5100001]


class Scope:
    def __init__(self, blocks, steps):
        self.blocks, self.steps = list(blocks), list(steps)

    def add(self, block, v):
        pass


class PadMass(Controller):
    def __init__(self, pipe, cfg, prompt, seeds):
        super().__init__(pipe, cfg, prompt, seeds, capture=Scope(range(models.N_BLOCKS['cogvideox']), PAD_STEPS))
        self.pad_index, self.mass = models.pad_index(pipe, self.model, prompt), {}

    @torch.no_grad()
    def attend(self, q, k, v, img, txt, act):
        pos = (txt.start+self.pad_index).to(q.device)
        qi, kt, scale = q[:, :, img], k.float().transpose(-1, -2), 1/math.sqrt(q.shape[-1])
        mass = torch.cat([(qi[:, :, c:c+CHUNK].float()@kt*scale).softmax(-1)[..., pos].sum(-1)
                          for c in range(0, qi.shape[2], CHUNK)], 2).mean(1)           # [B, n_video], mean over heads
        half = len(mass)//2                                                           # rows [uncond | cond]
        self.mass[f'{self.step}_{self.calls[-1][1]}'] = {
            h: dict(mean=t.mean().item(), share_gt_half=(t > .5).float().mean().item())
            for h, t in (('uncond', mass[:half]), ('cond', mass[half:]))}
        return super().attend(q, k, v, img, txt, act)


def peak(gpu):
    return torch.cuda.max_memory_allocated(gpu)/2**30


def cost(pipe, cfg, row, gpu):
    from generate import save_video                                           # scripts/generate.py
    emb, res, vids = models.encode(pipe, cfg.model, row['prompt']), dict(prompt_id=row['id'], calls=[]), {}
    folder = OUT/'cost_videos'
    folder.mkdir(parents=True, exist_ok=True)
    for batch in (1, 2):
        out = []
        for k in range(0, len(SEEDS), batch):
            torch.cuda.reset_peak_memory_stats(gpu)
            t0 = time.monotonic()
            out += models.generate(pipe, cfg.model, emb, SEEDS[k:k+batch])
            res['calls'].append(dict(batch=batch, seconds=time.monotonic()-t0, peak_gib=peak(gpu)))
            print(res['calls'][-1], flush=True)
        vids[batch] = out
    for s, v in zip(SEEDS, vids[1]):
        save_video(folder/f"{row['id']}_{s}.mp4", v, cfg.model.fps)
    d = np.abs(np.stack(vids[2]).astype(np.int16)-np.stack(vids[1]).astype(np.int16))
    res['batch2_vs_batch1'] = dict(same=bool((d == 0).all()), mean_abs=float(d.mean()), max_abs=int(d.max()))
    return res


def capture(pipe, cfg, row, gpu):
    steps = config.parse_range(cfg.method.v_edit.steps, cfg.model.steps)
    res = dict(prompt_id=row['id'], steps=[steps[0], steps[-1]], runs=[])
    for n in (4, 8):
        seeds = [cfg.data.calibration.seed_base+i for i in range(n)]
        torch.cuda.reset_peak_memory_stats(gpu)
        t0 = time.monotonic()
        try:
            basis.capture(pipe, cfg, [row['prompt']], seeds, range(models.N_BLOCKS['cogvideox']), steps)
            res['runs'].append(dict(seeds=n, seconds=time.monotonic()-t0, peak_gib=peak(gpu)))
        except torch.cuda.OutOfMemoryError:
            res['runs'].append(dict(seeds=n, oom=True))
            torch.cuda.empty_cache()
        print(res['runs'][-1], flush=True)
    return res


def padmass(pipe, cfg, rows):
    """Saves after every prompt and resumes from padmass.partial.json. An out-of-memory error ends the process (a retry
    in the same process breaks the CPU offload hooks); the caller restarts it."""
    part = OUT/'padmass.partial.json'
    res = json.loads(part.read_text()) if part.exists() else {}
    for version in ('long', 'short'):
        for r in rows:
            key = f"{version}/{r['id']}"
            if key in res:
                continue
            prompt = r['prompt'] if version == 'long' else r['short']
            ctrl = PadMass(pipe, cfg, prompt, SEEDS[:1])
            models.generate(pipe, cfg.model, models.encode(pipe, cfg.model, prompt), SEEDS[:1], ctrl,
                            last_step=max(PAD_STEPS))
            res[key] = dict(pad_tokens=len(ctrl.pad_index), mass=ctrl.mass)
            part.write_text(json.dumps(res))
            m = [v['cond']['mean'] for v in ctrl.mass.values()]
            print(version, r['id'], len(ctrl.pad_index), f'cond mean over blocks and steps {sum(m)/len(m):.4f}', flush=True)
    return res


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('command', choices=['cost', 'capture', 'padmass'])
    a.add_argument('--gpu', type=int, default=0)
    x = a.parse_args()
    cfg = config.load('cogvideox', 'original')
    rows = data.prompts(cfg.data, 'vbench_tune', samples=2)
    pipe = models.load(cfg.model, x.gpu)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if x.command == 'cost':
        res = cost(pipe, cfg, rows[0], x.gpu)
    elif x.command == 'capture':
        res = capture(pipe, cfg, rows[0], x.gpu)
    else:
        res = padmass(pipe, cfg, rows[:8])
    res['total_seconds'] = time.monotonic()-t0
    res['gpu'] = x.gpu
    (OUT/f'{x.command}.json').write_text(json.dumps(res, indent=1))
    print(f'wrote {OUT}/{x.command}.json', flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(config.ROOT/'scripts'))
    main()
