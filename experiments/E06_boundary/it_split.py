"""PROTOCOL_ko.md §22 (post-hoc): split the I->T change into spatial redistribution ("where") and concept flip ("what"),
and test which one tracks diversity and which one tracks the mismatch cost.

  python experiments/E06_boundary/it_split.py sd35     -> outputs/e06_boundary/<model>/e06/REPORT_itsplit_<model>.md

where: per concept token k, the patch distribution c_k = P[:, k] / sum_n P[n, k]; JS(c_k, c0_k) (base 2), weighted by
       Original's column mass share; block mean.
what:  share of patches whose most-attended concept token changed (signals.py it_flip); it_js reported alongside.
Unit: prompt x E1 condition (seed mean of 71000/71001). Test: z-score within condition across prompts, pool, OLS
outcome_z ~ where_z + what_z, prompt bootstrap (whole prompts resampled).
"""
import sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from analyze import load                                                      # noqa: E402
from signals import EPS, js                                                   # noqa: E402

B = 2000
LAST = dict(sd35=11, flux=5)


def comps(e, o, step, nb):
    keys = [f'{step},{b}' for b in range(nb)]
    pe = torch.stack([e['it'][k].float() for k in keys])                      # [nb, 256, nc]
    po = torch.stack([o['it'][k].float() for k in keys])
    # what: per-patch concept distribution
    qe, qo = pe/pe.sum(-1, keepdim=True).clamp_min(EPS), po/po.sum(-1, keepdim=True).clamp_min(EPS)
    flip = (qe.argmax(-1) != qo.argmax(-1)).float().mean(-1)                  # [nb]
    itjs = js(qe, qo).mean(-1)
    # where: per-concept patch distribution
    ce = (pe/pe.sum(-2, keepdim=True).clamp_min(EPS)).transpose(-1, -2)       # [nb, nc, 256]
    co = (po/po.sum(-2, keepdim=True).clamp_min(EPS)).transpose(-1, -2)
    w = po.sum(-2)/po.sum((-1, -2))[..., None].clamp_min(EPS)                 # [nb, nc] Original column mass share
    where = (js(ce, co)*w).sum(-1)
    return dict(where=float(where.mean()), what=float(flip.mean()), it_js=float(itjs.mean()))


def main():
    model = sys.argv[1]
    nb = 24 if model == 'sd35' else 57
    root = C.OUT/model/'e06'/'probe_sig'
    pids = sorted(p.name for p in root.iterdir() if p.is_dir())
    modes = sorted(p.name for p in (root/pids[0]).iterdir() if p.is_dir() and p.name != 'original')
    E1 = sorted([m for m in modes if m.startswith('v_l') and '+' not in m], key=lambda m: float(m[3:]))
    steps = (0, LAST[model])
    S = {}                                                                    # (pid, mode, step) -> dict
    for pid in pids:
        for s in C.SIGNAL_SEEDS:
            o = torch.load(root/pid/'original'/f'sig_seed{s}.pt', weights_only=True)
            for m in modes:
                f = root/pid/m/f'sig_seed{s}.pt'
                if not f.exists():
                    continue
                e = torch.load(f, weights_only=True)
                for st in steps:
                    d = comps(e, o, st, nb)
                    for k, v in d.items():
                        S.setdefault((pid, m, st), {}).setdefault(k, []).append(v)
    S = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in S.items()}
    T = load(model)
    outs = [('vendi', 'ΔVendi'), ('comp', '구도 다양성'), ('color', '색 다양성'), ('mismatch', '판독 불일치')]
    pids = [p for p in pids if (p, 'original') in T and all((p, m) in T for m in E1)]
    rng = np.random.default_rng(0)
    lines = [f'# E06 {model} I→T 변화의 어디/무엇 나누기 (PROTOCOL §22, 사후)', '',
             f'prompt {len(pids)}개, seed 2개 평균. 어디 = 개념 토큰별 패치 분포 JS(원본 열 질량 가중), 무엇 = 가장 많이 읽는 개념 토큰이 '
             '바뀐 패치 비율, it_js = 패치별 개념 분포 JS. 블록 평균.', '', '## 1. 조건별 평균 (원본 대비)', '',
             '| 조건 | step 0 어디 | step 0 무엇 | step 0 it_js | 마지막 편집 step 어디 | 무엇 | it_js |', '|---|---|---|---|---|---|---|']
    for m in E1+[m for m in modes if m not in E1]:
        c = [np.mean([S[(p, m, st)][k] for p in pids if (p, m, st) in S]) for st in steps for k in ('where', 'what', 'it_js')]
        lines.append(f'| {m} | '+' | '.join(f'{x:.4f}' for x in c)+' |')
    for st in steps:
        lines += ['', f'## 2. prompt 단위 시험 (step {st}, E1 {len(E1)}점, 조건 안 z-점수, prompt bootstrap {B})', '']
        X = {k: np.array([[S[(p, m, st)][k] for m in E1] for p in pids]) for k in ('where', 'what', 'it_js')}
        Y = {k: np.array([[T[(p, m)][k]-T[(p, 'original')][k] for m in E1] for p in pids]) for k, _ in outs}

        def z(a):
            return (a-a.mean(0))/(a.std(0)+1e-12)

        def fit(i, yk, wk='where', fk='what'):
            xw, xf, y = z(X[wk][i]).ravel(), z(X[fk][i]).ravel(), z(Y[yk][i]).ravel()
            A = np.c_[xw, xf]
            return np.linalg.lstsq(A, y, rcond=None)[0]
        r = np.corrcoef(z(X['where']).ravel(), z(X['what']).ravel())[0, 1]
        lines.append(f'- 어디·무엇 상관(조건 안 z): {r:.2f}')
        lines += ['', '| 결과 | 어디 계수 [95%] | 무엇 계수 [95%] | 어디 − 무엇 [95%] | 판정 |', '|---|---|---|---|---|']
        for yk, yn in outs:
            b0 = fit(np.arange(len(pids)), yk)
            bs = np.array([fit(rng.integers(0, len(pids), len(pids)), yk) for _ in range(B)])
            dif = bs[:, 0]-bs[:, 1]
            ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
            cw, cf, cd = ci(bs[:, 0]), ci(bs[:, 1]), ci(dif)
            if yk in ('vendi', 'comp'):
                ok = cw[0] > 0 and cd[0] > 0
                dec = '코멘트 지지' if ok else ('부분' if cw[0] > 0 or cd[0] > 0 else '지지하지 않음')
            elif yk == 'mismatch':
                ok = cf[0] > 0 and cd[1] < 0
                dec = '코멘트 지지' if ok else ('부분' if cf[0] > 0 or cd[1] < 0 else '지지하지 않음')
            else:
                dec = '기술용'
            lines.append(f'| {yn} | {b0[0]:+.2f} [{cw[0]:+.2f}, {cw[1]:+.2f}] | {b0[1]:+.2f} [{cf[0]:+.2f}, {cf[1]:+.2f}] | '
                         f'{b0[0]-b0[1]:+.2f} [{cd[0]:+.2f}, {cd[1]:+.2f}] | {dec} |')
        # it_js as the "what" measure (secondary)
        lines += ['', '보조: 무엇을 it_js로 바꾼 계수 (어디 / it_js)', '']
        for yk, yn in outs:
            b0 = fit(np.arange(len(pids)), yk, fk='it_js')
            lines.append(f'- {yn}: 어디 {b0[0]:+.2f}, it_js {b0[1]:+.2f}')
    rep = C.OUT/model/'e06'/f'REPORT_itsplit_{model}.md'
    rep.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
