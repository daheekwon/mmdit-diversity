"""Metrics of one video run (outputs/cogvideox/<dataset>/<tag>/videos/<prompt id>/<seed>.mp4), as fixed in
experiments/E02_t2v_cogvideox_pilot/PROTOCOL_ko.md section 6. Videos have 49 frames.

per video  subject_consistency  VBench subject consistency on DINOv2-small CLS features of all frames: mean over frames
                                i >= 1 of (cos(f_i, f_i-1) + cos(f_i, f_0)) / 2 (VBench uses DINO ViT-B/16)
           flicker_mae          mean absolute pixel difference of adjacent frames, 0-255 (VBench temporal flickering
                                before its (255 - x) / 255 rescaling; it also grows with motion)
           musiq                MUSIQ (pyiqa, KonIQ weights, as the image experiments) mean over frames 0, 6, ..., 48
           motion               VBench dynamic-degree pair score (mean of the largest 5% RAFT optical-flow magnitudes,
                                pixels) averaged over the 48 consecutive frame pairs (E02 PROTOCOL_ko.md section 7e)
           dynamic              VBench dynamic degree: 1 when at least round(4 x 49 / 16) = 12 of the pairs except the last
                                two have a pair score above 6 x (short side / 256)
           judge                Qwen3-VL-8B yes/no judge of the image experiments on frames 0, 16, 32, 48, with the
                                short original prompt: broken = any of the 4 frames has defect >= .5 or style >= .5;
                                mismatch = mean match probability of the 4 frames < .5
per prompt vendi_dino           Vendi (DINO kernel) over the prompt's videos; video feature = normalised mean of the
                                normalised features of frames 0, 6, ..., 48 (main diversity metric)
           vendi_dino_first     the same on frame 0 only (composition)
           vendi_dino_frames    mean over the 9 frame positions of the Vendi at that position
           and the means of the per-video metrics (broken, mismatch: shares).
DINO features (features.Embedder('dino'): whole frame, shorter side 224, no crop) and flicker are cached in
<run>/features/video_frames.npz, MUSIQ in <run>/features/musiq_frames.npz, judge answers in <run>/metrics/judge_frames.csv
(resumable), RAFT pair scores in <run>/features/flow_frames.npz. A cached metric that covers every finished video is
always reported.
Writes <run>/metrics/{per_video.csv, per_prompt.csv, summary.json}; scripts/compare.py reads the last two.
"""
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from . import features
from .metrics import vendi

N_FRAMES = 49
FRAMES9 = list(range(0, N_FRAMES, 6))
JUDGE_FRAMES = [0, 16, 32, 48]
METRICS = ('dino', 'musiq', 'judge', 'flow')
FLOW_TOP, FLOW_UPDATES, FLOW_BATCH = .05, 20, 8   # VBench dynamic degree: top 5% of the magnitudes, RAFT 20 iterations
ANAL = Path('/NHNHOME/workspace/diversity_anal/div-attn')    # MUSIQ runtime and weights, judge (image experiments)


def read_video(path):
    import imageio.v2 as imageio
    v = np.stack(imageio.mimread(path, memtest=False))
    if len(v) != N_FRAMES:
        raise ValueError(f'{path}: {len(v)} frames')
    return v


def run_rows(run):
    rows = [json.loads(line) for line in (Path(run)/'prompts.jsonl').read_text().splitlines()]
    for r in rows:
        r['files'] = [Path(run)/'videos'/r['id']/f'{s}.mp4' for s in r['seeds']]
    return rows


def video_feature(F, frames=FRAMES9):
    """F: [..., frames, d] normalised frame features -> normalised mean over `frames`."""
    m = F[..., frames, :].mean(-2)
    return m/np.linalg.norm(m, axis=-1, keepdims=True)


def subject_consistency(F):
    """F: [frames, d] normalised."""
    return float((((F[1:]*F[:-1]).sum(1)+(F[1:]*F[0]).sum(1))/2).mean())


def flicker_mae(v):
    """v: [frames, H, W, 3] uint8."""
    return float(np.abs(np.diff(v.astype(np.int16), axis=0)).mean())


def top_mean(rad, top=FLOW_TOP):
    """rad: [..., H, W] flow magnitudes (torch) -> mean of the largest int(H W top) values of each map (VBench get_score)."""
    flat = rad.reshape(*rad.shape[:-2], -1)
    return flat.topk(int(flat.shape[-1]*top), dim=-1).values.mean(-1)


def vbench_dynamic(scores, short_side, n_frames=N_FRAMES):
    """VBench dynamic degree of one video from its consecutive-pair scores (VBench check_move): 1 when at least
    round(4 n / 16) of the pairs except the last two score above 6 (short side / 256)."""
    thres, need = 6.*short_side/256., round(4*n_frames/16)
    return float(sum(s > thres for s in scores[:-2]) >= need)


def judge_video(j):
    """j: judge rows of one video (columns frame, p_match, p_defect, p_style) -> video-level answers."""
    return dict(broken=float(((j.p_defect >= .5) | (j.p_style >= .5)).any()), mismatch=float(j.p_match.mean() < .5),
                p_match=j.p_match.mean(), p_defect=j.p_defect.mean(), p_style=j.p_style.mean())


def _keys(run, files):
    return [str(f.relative_to(run)) for f in files]


def _load(path, keys):
    if path.exists():
        z = np.load(path)
        if list(z['files']) == keys:
            return {k: z[k] for k in z.files if k != 'files'}
    return None


def _save(path, keys, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')
    np.savez(tmp, files=np.array(keys), **arrays)
    os.replace(tmp, path)


def dino_frames(run, files, device, compute=True):
    """DINO features [videos, 49, 384] and flicker_mae [videos]."""
    from PIL import Image
    path, keys = run/'features'/'video_frames.npz', _keys(run, files)
    have = _load(path, keys)
    if have is not None or not compute:
        return have
    net, X, fl = features.Embedder('dino', device), [], []
    for f in files:
        v = read_video(f)
        X.append(np.concatenate([net([Image.fromarray(fr) for fr in v[i:i+25]]) for i in range(0, N_FRAMES, 25)]))
        fl.append(flicker_mae(v))
    del net
    torch.cuda.empty_cache()
    _save(path, keys, dino=np.stack(X), flicker=np.array(fl))
    return _load(path, keys)


def musiq_frames(run, files, device, compute=True):
    """MUSIQ [videos, 9] on frames 0, 6, ..., 48 at the generated size."""
    path, keys = run/'features'/'musiq_frames.npz', _keys(run, files)
    have = _load(path, keys)
    if have is not None or not compute:
        return have
    sys.path.insert(0, str(ANAL/'vendor'/'iqa_runtime'))
    os.environ['TORCH_HOME'] = str(ANAL/'model_cache'/'torch_eval')
    import pyiqa
    net, out = pyiqa.create_metric('musiq', device=device).eval(), []
    with torch.no_grad():
        for f in files:
            v = read_video(f)
            out.append([net(torch.from_numpy(v[i]).permute(2, 0, 1).float()[None].to(device)/255).flatten()[0].item()
                        for i in FRAMES9])
    del net
    torch.cuda.empty_cache()
    _save(path, keys, musiq=np.array(out))
    return _load(path, keys)


def flow_frames(run, files, device, compute=True):
    """RAFT large with the FlyingChairs + FlyingThings3D weights (torchvision C_T_V1, ported from the original raft-things
    weights that VBench uses), 20 iterations, full resolution: pair scores [videos, 48] and the short side per video."""
    path, keys = run/'features'/'flow_frames.npz', _keys(run, files)
    have = _load(path, keys)
    if have is not None or not compute:
        return have
    os.environ['TORCH_HOME'] = str(ANAL/'model_cache'/'torch_eval')
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    w = Raft_Large_Weights.C_T_V1
    net, tf, out, short = raft_large(weights=w, progress=False).eval().to(device), w.transforms(), [], []
    with torch.no_grad():
        for f in files:
            v = torch.from_numpy(read_video(f)).permute(0, 3, 1, 2)
            s = []
            for i in range(0, N_FRAMES-1, FLOW_BATCH):
                j = min(i+FLOW_BATCH, N_FRAMES-1)
                a, b = tf(v[i:j], v[i+1:j+1])
                flow = net(a.to(device), b.to(device), num_flow_updates=FLOW_UPDATES)[-1]
                s.append(top_mean(flow.float().norm(dim=1)).cpu())
            out.append(torch.cat(s).numpy())
            short.append(min(v.shape[-2:]))
    del net
    torch.cuda.empty_cache()
    _save(path, keys, top5=np.stack(out), short=np.array(short))
    return _load(path, keys)


def judge_frames(run, rows, device, compute=True):
    """Judge answers per (video, frame), appended to <run>/metrics/judge_frames.csv; None when incomplete."""
    from PIL import Image
    path = run/'metrics'/'judge_frames.csv'
    items = [(str(f.relative_to(run)), fr, r['short']) for r in rows for f in r['files'] for fr in JUDGE_FRAMES]
    have = pd.read_csv(path) if path.exists() else None
    done = set() if have is None else set(zip(have.video, have.frame))
    todo = [it for it in items if (it[0], it[1]) not in done]
    if todo and compute:
        sys.path.insert(0, str(ANAL))
        sys.path.insert(0, str(ANAL/'scripts'))
        from judge_flux_breakage import Judge, QUESTIONS
        j, cur, frames = Judge(device), None, None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['video', 'frame']+[f'p_{q}' for q in QUESTIONS])
            if have is None:
                w.writeheader()
            for k, (video, fr, prompt) in enumerate(todo):
                if video != cur:
                    cur, frames = video, read_video(run/video)
                im = Image.fromarray(frames[fr])
                w.writerow(dict(video=video, frame=fr, **{f'p_{q}': round(j(im, t.format(prompt=prompt)), 5)
                                                          for q, t in QUESTIONS.items()}))
                fh.flush()
                if k % 200 == 0:
                    print('judged', k, '/', len(todo), flush=True)
        del j
        torch.cuda.empty_cache()
        have = pd.read_csv(path)
    if have is None:
        return None
    have = have[have.video.isin({it[0] for it in items})]
    return have if len(set(zip(have.video, have.frame))) == len(items) else None


def evaluate(run, device='cuda:0', which=METRICS):
    run = Path(run)
    rows = run_rows(run)
    done = [r for r in rows if all(f.exists() for f in r['files'])]
    if len(done) < len(rows):
        print(f'{len(rows)-len(done)} prompts incomplete; evaluating {len(done)}', flush=True)
    files = [f for r in done for f in r['files']]
    pid = np.repeat([r['id'] for r in done], [len(r['files']) for r in done])
    vid = pd.DataFrame(dict(id=pid, file=_keys(run, files)))
    per = pd.DataFrame(index=pd.Index([r['id'] for r in done], name='id'))
    d = dino_frames(run, files, device, 'dino' in which)
    if d is not None:
        vid['subject_consistency'] = [subject_consistency(F) for F in d['dino']]
        vid['flicker_mae'] = d['flicker']
        per['vendi_dino'] = [vendi(video_feature(d['dino'][pid == i])) for i in per.index]
        per['vendi_dino_first'] = [vendi(d['dino'][pid == i][:, 0]) for i in per.index]
        per['vendi_dino_frames'] = [np.mean([vendi(d['dino'][pid == i][:, k]) for k in FRAMES9]) for i in per.index]
    m = musiq_frames(run, files, device, 'musiq' in which)
    if m is not None:
        vid['musiq'] = m['musiq'].mean(1)
    fl = flow_frames(run, files, device, 'flow' in which)
    if fl is not None:
        vid['motion'] = fl['top5'].mean(1)
        vid['dynamic'] = [vbench_dynamic(s, h) for s, h in zip(fl['top5'], fl['short'])]
    j = judge_frames(run, done, device, 'judge' in which)
    if j is not None:
        jv = pd.DataFrame([dict(file=f, **judge_video(g)) for f, g in j.groupby('video')]).set_index('file')
        vid = vid.join(jv, on='file')
    per = per.join(vid.drop(columns='file').groupby('id', sort=False).mean())
    summary = dict(run=str(run), prompts=len(done), videos=len(files), prompts_total=len(rows),
                   **{f'{c}_mean': float(per[c].mean()) for c in per.columns})
    out = run/'metrics'
    out.mkdir(exist_ok=True)
    vid.to_csv(out/'per_video.csv', index=False)
    per.to_csv(out/'per_prompt.csv')
    (out/'summary.json').write_text(json.dumps(summary, indent=1))
    return summary
