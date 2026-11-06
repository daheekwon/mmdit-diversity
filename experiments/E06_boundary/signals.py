"""E06 internal signals (PROTOCOL §4.3 H1/H2/H4/H5, predictions P4/P5; §9 capture) against Original, per image pair
(same prompt, same seed 71000/71001):

  it_js    I->T: mean over patches of the JS divergence (base 2) between the concept-token distributions (each patch's
           concept probabilities renormalised), head mean, 2x2-pooled patches
  it_flip  share of patches whose most-attended concept token changed
  it_mass  relative change of the image queries' total concept mass, (m - m0) / m0 (signed)
  kc_cos   cosine of concept-token keys (128-d projection) with Original, mean over tokens
  ln_sig   relative change of the block-input LayerNorm sigma of the image tokens, sigma / sigma0 - 1
  d        RMS z-score of the token-mean V-basis coordinates (edited conditions: lam * c at edited steps) against
           Original's distribution over all prompts and seeds at that step and block (not for the random-direction
           condition: its probe recorded coordinates in its own basis R)
  x0       ||x0_hat - x0_hat_orig|| / ||x0_hat_orig|| at the recorded steps
Block mean over all blocks (SD3.5 also blocks 0-5 alone). Intervals: prompt bootstrap of per-prompt means (2,000, seed 0).
Image-level link (for the E3 reading, PROTOCOL §5.1): Spearman between an image's signal and its judge flags over the E1
images, prompt bootstrap.

  python experiments/E06_boundary/signals.py sd35
"""
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402

B = 2000
EPS = 1e-8


def js(p, q):
    """p, q: [..., n] distributions -> JS divergence base 2 [...]."""
    m = .5*(p+q)
    kl = lambda a, b: (a*(torch.log2(a+EPS)-torch.log2(b+EPS))).sum(-1)
    return .5*kl(p, m)+.5*kl(q, m)


def lam_eff(cfg_mode, step, model):
    """Effective scale of the V-basis coordinates at a step: lam at edited steps for V-edit conditions (Q basis)."""
    if not cfg_mode.startswith('v_l'):
        return 1.0
    lam = float(cfg_mode[3:].split('+')[0])
    lo, hi = (0, 11) if model == 'sd35' else (0, 5)
    return lam if lo <= step <= hi else 1.0


def pair_metrics(e, o, steps, nb):
    out = {}
    for s in steps:
        keys = [f'{s},{b}' for b in range(nb)]
        pe = torch.stack([e['it'][k].float() for k in keys])                  # [nb, 256, nc]
        po = torch.stack([o['it'][k].float() for k in keys])
        me, mo = pe.sum(-1), po.sum(-1)
        qe, qo = pe/me[..., None].clamp_min(EPS), po/mo[..., None].clamp_min(EPS)
        j = js(qe, qo).mean(-1)                                              # [nb]
        flip = (qe.argmax(-1) != qo.argmax(-1)).float().mean(-1)
        mass = (me.mean(-1)-mo.mean(-1))/mo.mean(-1).clamp_min(EPS)
        ke = torch.stack([e['kc'][k].float() for k in keys])                  # [nb, nc, 128]
        ko = torch.stack([o['kc'][k].float() for k in keys])
        kcos = torch.nn.functional.cosine_similarity(ke, ko, dim=-1).mean(-1)
        lse = torch.stack([e['ln'][k] for k in keys])[:, 0]
        lso = torch.stack([o['ln'][k] for k in keys])[:, 0]
        out[s] = dict(it_js=j.numpy(), it_flip=flip.numpy(), it_mass=mass.numpy(), kc_cos=kcos.numpy(),
                      ln_sig=(lse/lso-1).numpy())
    x0 = {}
    for k in e['x0']:
        a, b = e['x0'][k].float(), o['x0'][k].float()
        x0[int(k)] = float((a-b).norm()/b.norm())
    return out, x0


def main():
    model = sys.argv[1]
    nb = 24 if model == 'sd35' else 57
    steps = C.SIGNAL_STEPS[model]
    root = C.OUT/model/'e06'/'probe_sig'
    pids = sorted(p.name for p in root.iterdir() if p.is_dir())
    modes = sorted(p.name for p in (root/pids[0]).iterdir() if p.is_dir() and p.name != 'original')
    seeds = C.SIGNAL_SEEDS
    load = lambda pid, m, s: torch.load(root/pid/m/f'sig_seed{s}.pt', weights_only=True)
    # pass 1: Original's coordinate distribution per (step, block) for d
    cb = {}
    for pid in pids:
        for s in seeds:
            o = load(pid, 'original', s)
            for k, v in o['c'].items():
                cb.setdefault(k, []).append(v[0].numpy())
    mu = {k: np.mean(v, 0) for k, v in cb.items()}
    sd = {k: np.std(v, 0)+1e-6 for k, v in cb.items()}
    # pass 2: pairs
    rows = []                         # (pid, seed, mode, metric, step, value-by-block or scalar)
    per = {}                          # per[(mode, metric, step)][pid] -> list over seeds of block-mean (and blocks 0-5)
    img = {}                          # img[(pid, mode, seed)][(metric, step)] = block mean
    for pid in pids:
        for s in seeds:
            o = load(pid, 'original', s)
            for m in modes:
                f = root/pid/m/f'sig_seed{s}.pt'
                if not f.exists():
                    continue
                e = torch.load(f, weights_only=True)
                mets, x0 = pair_metrics(e, o, steps, nb)
                rec = img.setdefault((pid, m, f'seed{s}'), {})
                for st, d in mets.items():
                    for met, vals in d.items():
                        per.setdefault((m, met, st), {}).setdefault(pid, []).append(float(vals.mean()))
                        rec[(met, st)] = float(vals.mean())
                        if model == 'sd35':
                            per.setdefault((m, met+'_b0-5', st), {}).setdefault(pid, []).append(float(vals[:6].mean()))
                    zs = []
                    if m.startswith('rnd'):                  # coordinates were recorded in the random basis R, not Q
                        continue
                    for b in range(nb):
                        k = f'{st},{b}'
                        if k in e['c']:
                            z = (lam_eff(m, st, model)*e['c'][k][0].numpy()-mu[k])/sd[k]
                            zs.append(float(np.sqrt((z**2).mean())))
                    if zs:
                        per.setdefault((m, 'd', st), {}).setdefault(pid, []).append(float(np.mean(zs)))
                        rec[('d', st)] = float(np.mean(zs))
                for st, v in x0.items():
                    per.setdefault((m, 'x0', st), {}).setdefault(pid, []).append(v)
                    rec[('x0', st)] = v
    rng = np.random.default_rng(0)

    def ci(d):
        v = np.array([np.mean(x) for x in d.values()])
        bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    E1 = sorted([m for m in modes if m.startswith('v_l') and '+' not in m], key=lambda m: float(m[3:]))
    E2 = [f'padv{f}' for f in ('0.5', '0.25', '0.125', '0') if f'padv{f}' in modes]
    other = [m for m in modes if m not in E1+E2]
    order = E1+E2+other
    out = [f'# E06 {model} 내부 신호 (원본 대비, prompt {len(pids)}개 × seed {len(seeds)}, block 평균, [95% 구간])', '']
    cols = [('it_js', 'I→T JS'), ('it_flip', '최대 토큰 뒤집힘'), ('it_mass', '개념 질량 변화'), ('kc_cos', '개념 K 코사인'),
            ('ln_sig', 'LN σ 변화'), ('d', 'd')]
    for st in steps:
        out += [f'## step {st}', '', '| 조건 | '+' | '.join(c[1] for c in cols)+' |', '|'+'---|'*(len(cols)+1)]
        for m in order:
            cells = []
            for met, _ in cols:
                d = per.get((m, met, st))
                cells.append('—' if not d else '{:+.4f} [{:+.4f}, {:+.4f}]'.format(*ci(d)))
            out.append(f'| {m} | '+' | '.join(cells)+' |')
        out.append('')
    if model == 'sd35':
        out += ['## SD3.5 블록 0–5만 (I→T JS)', '', '| 조건 | '+' | '.join(f'step {s}' for s in steps)+' |', '|'+'---|'*(len(steps)+1)]
        for m in order:
            out.append(f'| {m} | '+' | '.join('{:+.4f}'.format(ci(per[(m, 'it_js_b0-5', s)])[0]) for s in steps)+' |')
        out.append('')
    x0s = sorted({st for (m, met, st) in per if met == 'x0'})
    out += ['## x̂₀ 상대 거리', '', '| 조건 | '+' | '.join(f'step {s}' for s in x0s)+' |', '|'+'---|'*(len(x0s)+1)]
    for m in order:
        out.append(f'| {m} | '+' | '.join('{:.3f}'.format(ci(per[(m, 'x0', s)])[0]) if (m, 'x0', s) in per else '—' for s in x0s)+' |')
    # P4: step-0 I->T change monotone in lam over E1
    s0 = steps[0]
    p4 = {met: [ci(per[(m, met, s0)])[0] for m in E1] for met in ('it_js', 'it_flip')}
    mono = {met: bool(all(b > a for a, b in zip(v, v[1:]))) for met, v in p4.items()}
    out += ['', '## P4 (step 0의 I→T 변화가 λ에 단조 증가)', '']
    for met, v in p4.items():
        out.append(f'- {met}: '+', '.join(f'{m} {x:.4f}' for m, x in zip(E1, v))+f' → {"단조 증가 ✔" if mono[met] else "단조 아님 ✘"}')
    # P5 (sigma part): LN sigma decreases as pad shrinks (s = 1 -> 0.5 -> ... -> 0), at every recorded edited step
    out += ['', '## P5 σ 부분 (pad가 작을수록 블록 입력 LayerNorm σ 단조 감소)', '']
    for st in steps:
        v = [0.0]+[ci(per[(m, 'ln_sig', st)])[0] for m in E2]
        ok = all(b < a for a, b in zip(v, v[1:]))
        out.append(f'- step {st}: 원본 0, '+', '.join(f'{m} {x:+.4f}' for m, x in zip(E2, v[1:]))+f' → {"단조 감소 ✔" if ok else "단조 아님 ✘"}')
    # image-level link with judge flags (E1 images)
    J = {}
    for fj in sorted((C.OUT/model/'e06').glob('judge_e06_*of2.csv')):
        for r in csv.DictReader(fj.open()):
            J[(r['pid'], r['mode'], r['seed'])] = (float(float(r['p_defect']) >= .5 or float(r['p_style']) >= .5),
                                                   float(float(r['p_match']) < .5))
    out += ['', '## 이미지 단위 연결 (E1 이미지, Spearman, prompt bootstrap)', '',
            '깨짐이 step 0 신호와 같이 움직이면 E3가 답할 수 있는 범위, 이후 step에서만 커지면 E3가 못 잡는 범위(§5.1).', '']

    from scipy.stats import rankdata

    def spearman(a, b):                                   # average ranks: the judge flags are 0/1 with many ties
        ra, rb = rankdata(a), rankdata(b)
        return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')
    keys = [k for k in img if k[1] in E1 and k in J]
    byp = {}
    for k in keys:
        byp.setdefault(k[0], []).append(k)
    for flag, fi in (('망가짐', 0), ('불일치', 1)):
        for met in ('it_js', 'it_flip', 'x0'):
            for st in (steps if met != 'x0' else (0, 11, 27)):
                if (met, st) not in img[keys[0]]:
                    continue
                a = np.array([img[k][(met, st)] for k in keys])
                b = np.array([J[k][fi] for k in keys])
                r = spearman(a, b)
                ps = list(byp)
                bs = []
                for _ in range(400):
                    pick = [k for p in rng.choice(ps, len(ps)) for k in byp[p]]
                    bs.append(spearman(np.array([img[k][(met, st)] for k in pick]), np.array([J[k][fi] for k in pick])))
                within = [spearman(np.array([img[k][(met, st)] for k in keys if k[1] == m]),
                                   np.array([J[k][fi] for k in keys if k[1] == m])) for m in E1]
                out.append(f'- {flag} ~ {met} step {st}: ρ {r:+.3f} [{np.nanpercentile(bs, 2.5):+.3f}, {np.nanpercentile(bs, 97.5):+.3f}] (이미지 {len(keys)}); '
                           f'조건 안 ρ 평균 {np.nanmean(within):+.3f} (사후 참고: 강도 효과를 뺀 연결)')
    rep = C.OUT/model/'e06'/f'SIGNALS_{model}.md'
    rep.write_text('\n'.join(out)+'\n')
    print('\n'.join(out))
    print(json.dumps(dict(P4=mono)))


if __name__ == '__main__':
    main()
