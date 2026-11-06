"""E4 guard analysis (PROTOCOL_ko.md §20; rule fixed before the E4 result).

  python experiments/E06_boundary/analyze_e4.py flux

Paths (each block separately), all vs Original:
  unguarded: Original (0), padv0.25, v_l<lam>+padv0.25 for the E4 lambdas
  guarded:   Original, padv0.25+fix<x> (E3), v_l<lam>+padv0.25+fix<x>
Main measure: guarded cost - unguarded polyline cost at the guarded condition's mean dVendi (linear interpolation between
condition means; paired prompt bootstrap, the polyline redrawn in each resample), for broken and mismatch.
Main comparison point: the guarded lambda condition whose mean dVendi is closest to the unguarded final point.
Also: same-lambda dVendi ratio, window ends (point estimates, descriptive), MUSIQ flag, pixel change.
Writes outputs/e06_boundary/<model>/e06/REPORT_E4_<model>.md.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from analyze import B, MARGIN, boot_mean, delta, fmt, load, prompts_of, window_end   # noqa: E402
from analyze_e3 import pixdiff                                                # noqa: E402

KEYS = [('vendi', 'ΔVendi', 2), ('vendi_clip', 'ΔCLIP Vendi (보조)', 2), ('broken', '망가짐', 1), ('mismatch', '불일치', 1), ('defect', 'defect', 1),
        ('style', 'style', 1), ('musiq', 'MUSIQ', 2), ('vqa', 'VQA', 3)]
COSTS = [('broken', '망가짐'), ('mismatch', '불일치')]


def final_point(model):
    """The final setting on the combination path: the 'ours' preset with no override."""
    return next(k for k, (p, o) in C.modes(model).items() if p == 'ours' and not o)


def verdict(tb, tm):
    """§20: tb / tm = (estimate, lo, hi) of guarded - unguarded polyline for broken / mismatch."""
    if tb[1] > 0 or tm[1] > 0:
        return '같은 다양성에서 가드가 비용을 늘린다'
    if tb[2] < 0 or tm[2] < 0:
        only_mis = tm[2] < 0 and not tb[2] < 0
        return '같은 다양성에서 가드가 비용을 줄인다' + (' (불일치 하나로만 갈림; FLUX 불일치는 잡음이 큼, §16)' if only_mis else '')
    return '같은 다양성에서 차이 없음'


def report(T, model, plain, guarded, final, rng):
    """plain: unguarded path conditions in strength order (without Original); guarded: block -> path conditions."""
    allm = ['original']+plain+[m for g in guarded.values() for m in g]
    missing = [m for m in allm if not prompts_of(T, m)]
    if missing:
        raise SystemExit(f'not evaluated yet: {missing}')
    pids = sorted(set.intersection(*[set(prompts_of(T, m)) for m in allm]))
    rare = [p for p in pids if p.startswith('rare_')]
    out = [f'# E06 {model} E4 가드 (PROTOCOL §20, 원본 대비, prompt 짝 bootstrap 95%)', '',
           f'prompt {len(pids)}개 × seed 8. 조합 = 이미지 V + pad ¼(편집 step 0–5). 가드 = E3 차단(fixit = I→T 개념 열 고정, '
           'fixtxt = 텍스트 스트림 고정)을 조합의 편집 step에 넣은 것.', '']
    # 1. per-condition table
    out += ['## 1. 조건별 값', '', '| 조건 | '+' | '.join(k[1] for k in KEYS)+' | 희귀 망가짐 / 불일치 |',
            '|'+'---|'*(len(KEYS)+2)]
    S = {}
    rows = plain+[m for g in guarded.values() for m in g]
    for m in rows:
        cells = []
        for k, _, nd in KEYS:
            d, _ = delta(T, m, k, pids=pids)
            t = boot_mean(d, np.random.default_rng(1) if k == 'vendi_clip' else rng)
            S.setdefault(m, {})[k] = t[0]
            cells.append(fmt(t, nd)+(' 플래그' if k == 'musiq' and t[0] < -1 else ''))
        r = [float(delta(T, m, k, pids=rare)[0].mean()) for k in ('broken', 'mismatch')]
        out.append(f'| {m} | '+' | '.join(cells)+f' | {r[0]:+.1f} / {r[1]:+.1f} |')

    def means(i, ms, key):
        return np.array([np.mean([T[(pids[j], m)][key]-T[(pids[j], 'original')][key] for j in i]) for m in ms])

    # 2. main measure
    xf = S[final]['vendi']
    out += ['', '## 2. 같은 평균 ΔVendi에서 가드 없는 경로와의 비용 차이 (주 지표)', '',
            f'가드 없는 경로 = 원본(0, 0), {", ".join(plain)}의 조건 평균점을 잇는 꺾은선. 가드 조건의 평균 ΔVendi에서 선형 보간한 '
            f'값을 뺀다. 주 비교점 = 가드 λ 조건 중 평균 ΔVendi가 가드 없는 최종점 {final}(+{xf:.2f})에 가장 가까운 점.', '']
    xmax = max(S[m]['vendi'] for m in plain)
    verdicts = {}
    for x, path in guarded.items():
        lam_pts = path[1:]
        main_pt = min(lam_pts, key=lambda m: abs(S[m]['vendi']-xf))
        out += [f'### fix{x} (주 비교점 {main_pt})', '', '| 조건 | ΔVendi | 망가짐 차이 | 불일치 차이 | 비고 |', '|---|---|---|---|---|']
        for m in path:
            ts = {}
            for k, _ in COSTS:
                vals = []
                for b in range(B+1):
                    i = np.arange(len(pids)) if b == 0 else rng.integers(0, len(pids), len(pids))
                    cx, cy = np.r_[0., means(i, plain, 'vendi')], np.r_[0., means(i, plain, k)]
                    o = np.argsort(cx)
                    bx, by = means(i, [m], 'vendi')[0], means(i, [m], k)[0]
                    vals.append(by-np.interp(bx, cx[o], cy[o]))
                ts[k] = (vals[0], float(np.percentile(vals[1:], 2.5)), float(np.percentile(vals[1:], 97.5)))
            note = []
            if S[m]['vendi'] > xmax:
                note.append('가드 없는 경로 범위 밖(끝값 사용)')
            if m == main_pt:
                verdicts[x] = verdict(ts['broken'], ts['mismatch'])
                note.append('**주 비교점**')
            out.append(f'| {m} | {S[m]["vendi"]:+.2f} | {fmt(ts["broken"])} | {fmt(ts["mismatch"])} | {"; ".join(note)} |')
        out += ['', f'- 판정(fix{x}, {main_pt}): **{verdicts[x]}**', '']
    # 3. diversity at the same lambda
    out += ['## 3. 같은 λ에서의 다양성 (ΔVendi 가드 / 가드 없음)', '']
    for x, path in guarded.items():
        for g, p in zip(path, plain):
            Xg, Xp = means(np.arange(len(pids)), [g], 'vendi')[0], means(np.arange(len(pids)), [p], 'vendi')[0]
            rat = []
            for _ in range(B):
                i = rng.integers(0, len(pids), len(pids))
                rat.append(means(i, [g], 'vendi')[0]/means(i, [p], 'vendi')[0])
            cg, cp = means(np.arange(len(pids)), [g], 'vendi_clip')[0], means(np.arange(len(pids)), [p], 'vendi_clip')[0]
            out.append(f'- {g} / {p}: {Xg/Xp:.2f} [{np.percentile(rat, 2.5):.2f}, {np.percentile(rat, 97.5):.2f}]'
                       f'; 보조 CLIP Vendi {cg/cp:.2f}')
        top = max(S[m]['vendi'] for m in path)
        out.append(f'- fix{x} 가드 조건의 최대 평균 ΔVendi {top:+.2f} vs 가드 없는 최종점 {xf:+.2f}'
                   + (' → 가드로는 최종점의 다양성에 닿지 못한다' if top < xf else ''))
        out.append('')
    # 4. window ends (descriptive)
    out += ['## 4. 창이 끝나는 ΔVendi (점추정, 기술용, §12)', '']
    for name, path in [('가드 없음', plain)]+[(f'fix{x}', p) for x, p in guarded.items()]:
        for k, n in COSTS:
            r = [('original', 0., 0.)]+[(m, S[m]['vendi'], S[m][k]) for m in path]
            xe, me = window_end(r, k)
            out.append(f'- {name} {n}: ' + ('범위 끝까지 창 안' if xe is None else f'ΔVendi {xe:.2f}에서 +{MARGIN:.0f}pp를 넘음 '
                                                                         f'(넘는 첫 조건 {me})'))
    # 5. pixel change
    out += ['', '## 5. 가드가 바꾼 픽셀 (같은 prompt·seed, 같은 λ 가드 없음 대비, 평균 |차| / 255, 처음 80장)', '']
    for x, path in guarded.items():
        for g, p in zip(path[1:], plain[1:]):
            v, n = pixdiff(model, p, g)
            out.append(f'- {g} vs {p}: {v:.1f} (n {n})')
    return out, verdicts


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'flux'
    rng = np.random.default_rng(0)
    T = load(model)
    lam = C.E4_LAM[model]
    plain = ['padv0.25']+[f'v_l{v}+padv0.25' for v in lam]
    guarded = {x: [f'padv0.25+fix{x}']+[f'v_l{v}+padv0.25+fix{x}' for v in lam] for x in C.E4_FIX[model]}
    out, verdicts = report(T, model, plain, guarded, final_point(model), rng)
    rep = C.gen_root(model, 'e06').parent/f'REPORT_E4_{model}.md'
    rep.write_text('\n'.join(out)+'\n')
    print('\n'.join(out))
    print(rep, verdicts)


if __name__ == '__main__':
    main()
