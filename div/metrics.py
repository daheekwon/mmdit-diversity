"""Metrics of one generation run (outputs/<model>/<dataset>/<tag>/).

per image   clip_score     100 x cos(image, text), CLIP ViT-B/32 (as the DAVE paper)
            pickscore      PickScore logit (logit_scale x cos)
            aesthetic      LAION aesthetic predictor v2 (MLP on CLIP ViT-L/14, weights/sac_logos_ava1_l14_linearMSE.pth)
            image_reward   ImageReward-v1.0 (package `image-reward`)
per prompt  means of the per-image scores; vendi_clip (CLIP ViT-B/32 kernel, as the DAVE paper), vendi_dino (DINOv2
            kernel, main diversity metric), lpips (mean over image pairs, AlexNet)
per run     fid, precision, recall, density, coverage: Inception pool3 against the dataset's real set
            (PRDC: Naeem et al. 2020, k = 5); computed only when every prompt of the dataset is complete.
"""
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import data, features
from .config import ROOT

PER_IMAGE = ('clip_score', 'pickscore', 'aesthetic', 'image_reward')
PER_PROMPT = ('vendi_clip', 'vendi_dino', 'lpips')
SET = ('fid', 'prdc')
AESTHETIC = ROOT/'weights'/'sac_logos_ava1_l14_linearMSE.pth'
PRDC_K = 5


def vendi(X):
    """exp(entropy of the eigenvalues of K/n), K the cosine kernel of the rows of X (Friedman & Dieng 2023)."""
    X = X/np.linalg.norm(X, axis=1, keepdims=True)
    lam = np.linalg.eigvalsh(X@X.T/len(X)).clip(0)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam*np.log(lam)).sum()))


def fid(a, b):
    from scipy import linalg
    ma, mb, ca, cb = a.mean(0), b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s, _ = linalg.sqrtm(ca@cb, disp=False)
    return float(((ma-mb)**2).sum()+np.trace(ca)+np.trace(cb)-2*np.trace(np.real(s)))


def prdc(real, fake, k=PRDC_K):
    """Precision / recall / density / coverage (NAVER prdc reference implementation, k-th neighbour excludes self)."""
    d = lambda a, b: torch.cdist(torch.from_numpy(a).double(), torch.from_numpy(b).double())
    kth = lambda m: m.kthvalue(k+1, dim=1).values
    rr, ff, rf = kth(d(real, real)), kth(d(fake, fake)), d(real, fake)
    return dict(precision=float((rf < rr[:, None]).any(0).double().mean()),
                recall=float((rf < ff[None, :]).any(1).double().mean()),
                density=float((rf < rr[:, None]).sum(0).double().mean()/k),
                coverage=float((rf.min(1).values < rr).double().mean()))


class Aesthetic:
    def __init__(self, device):
        sd = torch.load(AESTHETIC, weights_only=True, map_location='cpu')
        dims = [768, 1024, 128, 64, 16, 1]
        layers = [torch.nn.Linear(a, b) for a, b in zip(dims, dims[1:])]
        for layer, j in zip(layers, [0, 2, 4, 6, 7]):                  # the original model's dropout slots skipped
            layer.weight.data, layer.bias.data = sd[f'layers.{j}.weight'], sd[f'layers.{j}.bias']
        self.mlp = torch.nn.Sequential(*layers).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, emb):
        return self.mlp(torch.from_numpy(emb).to(self.device)).squeeze(-1).cpu().numpy()


def image_reward(device):
    """ImageReward-v1.0. Package 1.5 was written for transformers < 4.40 and setuptools < 70: put back the three helpers
    it imports from transformers.modeling_utils (now in transformers.pytorch_utils) and pkg_resources.packaging."""
    import packaging.version
    import pkg_resources
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    pkg_resources.packaging = packaging
    for f in ('apply_chunking_to_forward', 'find_pruneable_heads_and_indices', 'prune_linear_layer'):
        if not hasattr(mu, f):
            setattr(mu, f, getattr(pu, f))
    import ImageReward as RM
    return RM.load('ImageReward-v1.0', device=device)


def run_rows(run):
    rows = [json.loads(line) for line in (Path(run)/'prompts.jsonl').read_text().splitlines()]
    for r in rows:
        r['files'] = [Path(run)/'images'/r['id']/f'{s}.png' for s in r['seeds']]
    return rows


def evaluate(run, cfg, device='cuda:0', which=PER_IMAGE+PER_PROMPT+SET):
    run = Path(run)
    rows = run_rows(run)
    done = [r for r in rows if all(f.exists() for f in r['files'])]
    if len(done) < len(rows):
        print(f'{len(rows)-len(done)} prompts incomplete; evaluating {len(done)}', flush=True)
    files = [f for r in done for f in r['files']]
    pid = np.repeat([r['id'] for r in done], [len(r['files']) for r in done])
    img = pd.DataFrame(dict(id=pid, file=[str(f.relative_to(run)) for f in files]))
    texts = [r['prompt'] for r in done for _ in r['files']]
    feats = {}

    def feat(name):
        if name not in feats:
            feats[name] = features.cached(run, name, files, device)
        return feats[name]
    if 'clip_score' in which:
        net = features.Embedder('clip', device)
        feats['clip'] = features.cached(run, 'clip', files, device, model=net)
        t = net.text([r['prompt'] for r in done])
        img['clip_score'] = 100*np.clip((feats['clip']*np.repeat(t, [len(r['files']) for r in done], 0)).sum(1), 0, None)
    if 'pickscore' in which:
        net = features.Embedder('pick', device)
        x = features.cached(run, 'pick', files, device, model=net)
        t = np.repeat(net.text([r['prompt'] for r in done]), [len(r['files']) for r in done], 0)
        img['pickscore'] = net.logit_scale*(x*t).sum(1)
    if 'aesthetic' in which:
        img['aesthetic'] = Aesthetic(device)(feat('clip_l14'))
    if 'image_reward' in which:
        net = image_reward(device)
        with torch.no_grad():
            img['image_reward'] = [s for r in done for s in np.atleast_1d(net.score(r['prompt'], [str(f) for f in r['files']]))]
    per = img.drop(columns='file').groupby('id', sort=False).mean()
    if 'vendi_clip' in which:
        per['vendi_clip'] = [vendi(feat('clip')[pid == i]) for i in per.index]
    if 'vendi_dino' in which:
        per['vendi_dino'] = [vendi(feat('dino')[pid == i]) for i in per.index]
    if 'lpips' in which:
        per['lpips'] = _lpips(done, device)
    summary = dict(run=str(run), prompts=len(done), images=len(files), prompts_total=len(rows),
                   **{f'{c}_mean': float(per[c].mean()) for c in per.columns})
    if len(done) == len(rows) and any(s in which for s in SET):
        dataset, size = json.loads((run/'run.json').read_text())['dataset'], cfg.model.resolution
        real_dir = ROOT/'outputs'/'real'/f'{data.base(cfg.data, dataset)}_{size}'
        real = features.cached(real_dir, 'inception', data.real_images(cfg.data, dataset), device, size=size)
        fake = feat('inception')
        summary.update(n_real=len(real), n_fake=len(fake))
        if 'fid' in which:
            summary['fid'] = fid(real, fake)
        if 'prdc' in which and len(fake) > PRDC_K:                # k-th neighbour among the generated images
            summary.update(prdc(real, fake))
    out = run/'metrics'
    out.mkdir(exist_ok=True)
    img.to_csv(out/'per_image.csv', index=False)
    per.to_csv(out/'per_prompt.csv')
    (out/'summary.json').write_text(json.dumps(summary, indent=1))
    return summary


@torch.no_grad()
def _lpips(rows, device):
    import lpips
    net = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    out = []
    for r in rows:
        x = torch.stack([torch.from_numpy(np.array(features.load_image(f))).permute(2, 0, 1) for f in r['files']])
        x = (x.float()/127.5-1).to(device)
        i, j = zip(*combinations(range(len(x)), 2))
        out.append(float(net(x[list(i)], x[list(j)]).mean()))
    return out
