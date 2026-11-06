"""E3 feedback-blocking analysis (PROTOCOL_ko.md §5 "E3 판정", §3, §14).

  python experiments/E06_boundary/analyze_e3.py sd35

For each block (fixit = image->text concept columns fixed to the unedited twin, fixtxt = text stream fixed to the twin):
- cost slope through the origin (prompt x condition points, x = dVendi, y = d cost in pp, both vs Original), blocked vs
  edit-only on the SAME lambda points (paired prompt bootstrap of the difference); E1 all-point slope as a reference;
- diversity at the same lambda: dVendi(blocked) / dVendi(edit-only);
- pad point: one condition each, so the "slope" is cost / dVendi at that point;
- §5 supplementary rule: blocked dVendi below the E1 lowest named point (SD3.5 1.07, FLUX 1.43) -> add one lambda;
- pixel change of the blocked image against the edit-only image (same prompt and seed), to show the block acts.
Writes outputs/e06_boundary/<model>/e06/REPORT_E3_<model>.md.
"""
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from analyze import B, boot_mean, delta, fmt, load, prompts_of               # noqa: E402

LOWEST = dict(sd35='v_l1.07', flux='v_l1.43')          # §5 / §10 supplementary rule names these points
COSTS = [('broken', '망가짐'), ('mismatch', '불일치'), ('defect', 'defect'), ('style', 'style')]


def slope(X, Y):
    x, y = X.ravel(), Y.ravel()
    return float((x*y).sum()/(x*x).sum())


def mat(T, pids, modes, key):
    return np.array([[T[(p, m)][key]-T[(p, 'original')][key] for m in modes] for p in pids])


def compare(T, pids, edit, blocked, key, rng):
    """Slope difference blocked - edit-only and the diversity ratio, paired prompt bootstrap."""
    Xe, Xb = mat(T, pids, edit, 'vendi'), mat(T, pids, blocked, 'vendi')
    Ye, Yb = mat(T, pids, edit, key), mat(T, pids, blocked, key)
    se, sb = slope(Xe, Ye), slope(Xb, Yb)
    ds, rat = [], []
    for _ in range(B):
        i = rng.integers(0, len(pids), len(pids))
        ds.append(slope(Xb[i], Yb[i])-slope(Xe[i], Ye[i]))
        rat.append(Xb[i].mean(0)/Xe[i].mean(0))
    rat = np.array(rat)
    return dict(se=se, sb=sb, d=(sb-se, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))),
                ratio=[(float(Xb[:, j].mean()/Xe[:, j].mean()), float(np.percentile(rat[:, j], 2.5)),
                        float(np.percentile(rat[:, j], 97.5))) for j in range(len(edit))])


def clip_ratio(T, pids, edit, blocked, rng):
    """Auxiliary (11-06): dCLIP-Vendi(blocked) / dCLIP-Vendi(edit-only), paired prompt bootstrap."""
    e = np.array([T[(p, edit)]['vendi_clip']-T[(p, 'original')]['vendi_clip'] for p in pids])
    b = np.array([T[(p, blocked)]['vendi_clip']-T[(p, 'original')]['vendi_clip'] for p in pids])
    bs = [b[i].mean()/e[i].mean() for i in (rng.integers(0, len(pids), len(pids)) for _ in range(B))]
    return float(b.mean()/e.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pixdiff(model, a, b, n=80):
    """Mean absolute pixel difference (0-255) between conditions a and b over the first n (prompt, seed) pairs."""
    gen = C.gen_root(model, 'e06')
    out = []
    for p in sorted(x.name for x in gen.iterdir())[:n//8+1]:
        for s in C.SEEDS['e06']:
            fa, fb = gen/p/a/f'seed{s}.png', gen/p/b/f'seed{s}.png'
            if fa.exists() and fb.exists():
                ia = np.asarray(Image.open(fa).convert('RGB'), dtype=float)
                ib = np.asarray(Image.open(fb).convert('RGB'), dtype=float)
                out.append(np.abs(ia-ib).mean())
    return float(np.mean(out[:n])), len(out[:n])


def main():
    model = sys.argv[1]
    rng = np.random.default_rng(0)
    rng_c = np.random.default_rng(1)      # auxiliary CLIP Vendi: own stream, so the main intervals do not move
    T = load(model)
    allmodes = sorted({m for (_, m) in T})
    V = C.E3_V[model]
    pad = next(m.split('+')[0] for m in allmodes if m.startswith('padv') and m.endswith('+fixit'))
    E1 = sorted([m for m in allmodes if m.startswith('v_l') and '+' not in m], key=lambda m: float(m[3:]))
    e3modes = [f'{b}+fix{x}' for b in V+[pad] for x in ('it', 'txt')]
    pids = sorted(set.intersection(*[set(prompts_of(T, m)) for m in ['original']+E1+[pad]+e3modes]))
    rare = [p for p in pids if p.startswith('rare_')]
    out = [f'# E06 {model} E3 되먹임 차단 (PROTOCOL §5, 원본 대비, prompt 짝 bootstrap 95%)', '',
           f'prompt {len(pids)}개 × seed 8. 차단: fixit = I→T 개념 열 고정, fixtxt = 텍스트 스트림 고정(편집 step 0–'
           f'{11 if model == "sd35" else 5}, 같은 x_t의 편집 없는 짝 forward 값).', '']
    # 1. per-condition table
    keys = [('vendi', 'ΔVendi', 2), ('vendi_clip', 'ΔCLIP Vendi (보조)', 2), ('broken', '망가짐', 1), ('mismatch', '불일치', 1),
            ('defect', 'defect', 1), ('style', 'style', 1), ('musiq', 'MUSIQ', 2), ('vqa', 'VQA', 3)]
    out += ['## 1. 조건별 값', '', '| 조건 | '+' | '.join(k[1] for k in keys)+' | 희귀 망가짐 / 불일치 |', '|'+'---|'*(len(keys)+2)]
    S = {}
    for m in [LOWEST[model]]+[x for b in V+[pad] for x in (b, b+'+fixit', b+'+fixtxt')]:
        cells = []
        for k, _, nd in keys:
            d, _ = delta(T, m, k, pids=pids)
            t = boot_mean(d, rng_c if k == 'vendi_clip' else rng)
            S.setdefault(m, {})[k] = t[0]
            cells.append(fmt(t, nd))
        r = [float(delta(T, m, k, pids=rare)[0].mean()) for k in ('broken', 'mismatch')]
        out.append(f'| {m} | '+' | '.join(cells)+f' | {r[0]:+.1f} / {r[1]:+.1f} |')
    # 2. slopes
    out += ['', '## 2. 같은 Vendi에서의 비용 기울기 (원점 직선, pp/Vendi)', '',
            '편집만의 기울기는 같은 λ 점(차단과 짝)으로 맞춘다. E1 7점 전체 기울기는 참고값이다.', '']
    ref = {k: slope(mat(T, pids, E1, 'vendi'), mat(T, pids, E1, k)) for k, _ in COSTS}
    out.append('- E1 7점 전체 기울기(참고): '+', '.join(f'{n} {ref[k]:+.1f}' for k, n in COSTS))
    res = {}
    out += ['', '| 차단 | 비용 | 편집만 기울기 | 차단 기울기 | 차이 [95%] | 판정 |', '|---|---|---|---|---|---|']
    for x in ('it', 'txt'):
        for base, label in ((V, '이미지 V ' + '·'.join(V)), ([pad], pad)):
            for k, n in COSTS:
                r = compare(T, pids, base, [b+f'+fix{x}' for b in base], k, rng)
                res[(x, label, k)] = r
                dec = '줄어듦' if r['d'][2] < 0 else ('늘어남' if r['d'][1] > 0 else '차이 없음')
                out.append(f'| fix{x} | {label} {n} | {r["se"]:+.1f} | {r["sb"]:+.1f} | {fmt(r["d"])} | {dec} |')
    # 3. diversity
    out += ['', '## 3. 같은 λ에서의 다양성 (ΔVendi 차단 / 편집만)', '']
    for x in ('it', 'txt'):
        for base, label in ((V, '이미지 V ' + '·'.join(V)), ([pad], pad)):
            r = res[(x, label, 'broken')]
            for b, t in zip(base, r['ratio']):
                c = clip_ratio(T, pids, b, f'{b}+fix{x}', rng_c)
                out.append(f'- fix{x} {b}: {t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]' + (' (50% 미만)' if t[0] < .5 else '')
                           + f'; 보조 CLIP Vendi {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]')
    # 4. supplementary rule
    low = S[LOWEST[model]]['vendi']
    out += ['', f'## 4. 보충 규칙 (§5): 차단 조건 ΔVendi < E1 {LOWEST[model]} ΔVendi ({low:+.2f})', '']
    for m in e3modes:
        out.append(f'- {m}: {S[m]["vendi"]:+.2f} → ' + ('λ 한 점 추가 대상' if S[m]['vendi'] < low else '해당 없음'))
    # 5. pixel change
    out += ['', '## 5. 차단이 바꾼 픽셀 (같은 prompt·seed, 편집만 대비, 평균 |차| / 255, 처음 80장)', '',
            '참고: 반올림 수준 차이가 28 step을 거친 잡음 바닥은 평균 약 3/255(§9).', '']
    for m in e3modes:
        b = m.split('+fix')[0]
        v, n = pixdiff(model, b, m)
        vo, _ = pixdiff(model, 'original', b)
        out.append(f'- {m} vs {b}: {v:.1f} (n {n}); 참고 {b} vs 원본 {vo:.1f}')
    # 6. post-hoc: blocked point against the edit-only curve interpolated at the same mean dVendi
    out += ['', '## 6. 사후 확인 (사전 등록 아님): 같은 평균 ΔVendi에서 편집만 곡선과의 차이', '',
            '편집만 곡선 = 원본(0, 0)과 E1(이미지 V) 또는 E2(pad) 조건 평균점을 잇는 꺾은선. 차단 조건의 평균 ΔVendi에서 '
            '그 곡선의 비용을 선형 보간해 뺀다. 프롬프트 단위 원점 직선은 프롬프트 사이 이질성에 끌릴 수 있어 같이 본다.', '']
    E2 = [f'padv{f}' for f in ('0.5', '0.25', '0.125', '0')]

    def means(i, ms, key):
        return np.array([np.mean([T[(pids[j], m)][key]-T[(pids[j], 'original')][key] for j in i]) for m in ms])

    for m in e3modes:
        curve = E1 if m.startswith('v_l') else E2
        for k, n in COSTS[:2]:
            vals = []
            for b in range(B+1):
                i = np.arange(len(pids)) if b == 0 else rng.integers(0, len(pids), len(pids))
                cx, cy = np.r_[0., means(i, curve, 'vendi')], np.r_[0., means(i, curve, k)]
                o = np.argsort(cx)
                bx, by = means(i, [m], 'vendi')[0], means(i, [m], k)[0]
                vals.append(by-np.interp(bx, cx[o], cy[o]))
            t = (vals[0], float(np.percentile(vals[1:], 2.5)), float(np.percentile(vals[1:], 97.5)))
            out.append(f'- {m} {n}: 차단 − 편집만 곡선 {fmt(t)} pp (ΔVendi {S[m]["vendi"]:+.2f})')
    # 7. §17: DAVE standard setting with the same blocks (added after the SD3.5 E3 result)
    dave = C.E3_DAVE[model]
    dmodes = [f'{dave}+fix{x}' for x in ('it', 'txt')]
    if all(m in allmodes for m in dmodes):
        dp = sorted(set(pids) & set.intersection(*[set(prompts_of(T, m)) for m in [dave]+dmodes]))
        out += ['', f'## 7. DAVE 표준 설정 + 같은 차단 (§17, 결과를 본 뒤 추가한 조건)', '',
                f'prompt {len(dp)}개. 남는 비율 = ΔVendi(차단) / ΔVendi(편집만), 원본 대비. 차이 = DAVE − 우리(같은 재표집).', '']
        out += ['| 조건 | ΔVendi | ΔCLIP Vendi (보조) | 망가짐 | 불일치 | MUSIQ |', '|---|---|---|---|---|---|']
        for m in [dave]+dmodes:
            out.append(f'| {m} | '+' | '.join(fmt(boot_mean(delta(T, m, k, pids=dp)[0], rng_c if k == 'vendi_clip' else rng), nd) for k, nd in
                                              (('vendi', 2), ('vendi_clip', 2), ('broken', 1), ('mismatch', 1), ('musiq', 2)))+' |')
        out += ['', '차단 − DAVE(같은 prompt·seed 짝):', '']
        for m in dmodes:
            out.append(f'- {m}: '+', '.join(f'{n} {fmt(boot_mean(np.array([T[(p, m)][k]-T[(p, dave)][k] for p in dp]), rng), nd)}'
                                            for k, n, nd in (('broken', '망가짐', 1), ('mismatch', '불일치', 1), ('musiq', 'MUSIQ', 2))))
        dv = lambda m: np.array([T[(p, m)]['vendi']-T[(p, 'original')]['vendi'] for p in dp])
        out += ['', '| 차단 | DAVE 남는 비율 | 우리 λ | 우리 남는 비율 | 차이 DAVE − 우리 [95%] | 판정 |', '|---|---|---|---|---|---|']
        for x in ('it', 'txt'):
            Dd, Db = dv(dave), dv(f'{dave}+fix{x}')
            for lam in reversed(V):                                          # λ1.2 (main) first, then λ1.14
                Oe, Ob = dv(lam), dv(f'{lam}+fix{x}')
                rd, ro = Db.mean()/Dd.mean(), Ob.mean()/Oe.mean()
                bs = []
                for _ in range(B):
                    i = rng.integers(0, len(dp), len(dp))
                    bs.append(Db[i].mean()/Dd[i].mean()-Ob[i].mean()/Oe[i].mean())
                t = (rd-ro, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
                dec = ('우리가 더 잃음' if t[1] > 0 else '차이 없음/DAVE가 더 잃음') if lam == V[-1] else '(참고)'
                out.append(f'| fix{x} | {rd:.2f} | {lam} | {ro:.2f} | {fmt(t, 2)} | {dec} |')
        out += ['', '보조: 같은 비교를 CLIP Vendi로 (기술용, 판정에 쓰지 않음)', '']
        dc = lambda m: np.array([T[(p, m)]['vendi_clip']-T[(p, 'original')]['vendi_clip'] for p in dp])
        for x in ('it', 'txt'):
            for lam in reversed(V):
                Dd, Db, Oe, Ob = dc(dave), dc(f'{dave}+fix{x}'), dc(lam), dc(f'{lam}+fix{x}')
                bs = []
                for _ in range(B):
                    i = rng_c.integers(0, len(dp), len(dp))
                    bs.append(Db[i].mean()/Dd[i].mean()-Ob[i].mean()/Oe[i].mean())
                out.append(f'- fix{x} {lam}: DAVE {Db.mean()/Dd.mean():.2f}, 우리 {Ob.mean()/Oe.mean():.2f}, 차이 '
                           f'{fmt((Db.mean()/Dd.mean()-Ob.mean()/Oe.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), 2)}')
    # 8. DESIGN_e3_split_ko.md A: text route split at the image-query read (added after the SD3.5 E3 result)
    lam = C.SPLIT_LAM[model]
    smodes = [f'{lam}+{s}' for s in C.SPLIT]
    if all(m in allmodes for m in smodes):
        out += split_section(T, model, pids, lam, smodes, [k for k in keys if k[0] != 'vendi_clip'], rng)
    rep = C.gen_root(model, 'e06').parent/f'REPORT_E3_{model}.md'
    rep.write_text('\n'.join(out)+'\n')
    print('\n'.join(out))


SPLIT_LABEL = dict(readkv='텍스트 K·V 전부(텍스트 스트림 고정과 같은 조작)', readv='텍스트 V만', readk='텍스트 K만',
                   readwkv='단어 토큰 K·V만')


def split_section(T, model, pids, lam, smodes, keys, rng):
    """§8: the four read-level blocks against edit-only at one lam (DESIGN_e3_split_ko.md A, rules fixed before data)."""
    sp = sorted(set(pids) & set.intersection(*[set(prompts_of(T, m)) for m in smodes]))
    rare = [p for p in sp if p.startswith('rare_')]
    fixtxt = f'{lam}+fixtxt'
    out = ['', f'## 8. 텍스트 경로 쪼개기 (DESIGN_e3_split_ko.md A, 결과를 본 뒤 추가한 조건)', '',
           f'prompt {len(sp)}개 × seed 8, {lam}. 이미지 쿼리가 정해진 텍스트 열의 K·V를 짝 계산 값으로 읽는다(두 CFG 행).', '']
    out += ['| 조건 | '+' | '.join(k[1] for k in keys)+' | 희귀 망가짐 / 불일치 |', '|'+'---|'*(len(keys)+2)]
    for m in [lam, fixtxt]+smodes:
        cells = [fmt(boot_mean(delta(T, m, k, pids=sp)[0], rng), nd) for k, _, nd in keys]
        r = [float(delta(T, m, k, pids=rare)[0].mean()) for k in ('broken', 'mismatch')]
        name = m if m in (lam, fixtxt) else f'{m} ({SPLIT_LABEL[m.split("+")[1]]})'
        out.append(f'| {name} | '+' | '.join(cells)+f' | {r[0]:+.1f} / {r[1]:+.1f} |')
    out += ['', '**판정 (§5 규칙, 편집만 대비).** 망가짐 기울기 차이 구간 < 0 → 깨짐 줄어듦. ΔVendi 비 < 0.5 → 다양성도 줄어듦. '
            '"깨짐만 줄어듦"은 ho4에서 같은 규칙으로 다시 통과해야 가드 후보다.', '',
            '| 차단 | 망가짐 기울기 차이 [95%] | 불일치 기울기 차이 [95%] | ΔVendi 비 [95%] | 판정 |', '|---|---|---|---|---|']
    for m in smodes:
        rb = compare(T, sp, [lam], [m], 'broken', rng)
        rm = compare(T, sp, [lam], [m], 'mismatch', rng)
        ratio = rb['ratio'][0]
        cut = rb['d'][2] < 0
        div = ratio[0] >= .5
        dec = {(True, True): '깨짐만 줄어듦 (ho4 확인 필요)', (True, False): '다양성도 줄어듦',
               (False, True): '그대로', (False, False): '다양성만 줄어듦'}[(cut, div)]
        if rb['d'][1] > 0:
            dec += ' (망가짐 기울기 늘어남)'
        out.append(f'| {SPLIT_LABEL[m.split("+")[1]]} | {fmt(rb["d"])} | {fmt(rm["d"])} | '
                   f'{ratio[0]:.2f} [{ratio[1]:.2f}, {ratio[2]:.2f}] | {dec} |')
    # shares of the full read-level block (descriptive)
    out += ['', '**몫 (기술 통계, 판정 아님).** 몫 = (편집만 − 차단) / (편집만 − 텍스트 K·V 전부), 조건 평균, prompt bootstrap.', '',
            '| 차단 | 망가짐 몫 | ΔVendi 몫 |', '|---|---|---|']
    full = f'{lam}+readkv'
    dd = lambda m, k: np.array([T[(p, m)][k]-T[(p, 'original')][k] for p in sp])
    shares = {}
    for k in ('broken', 'vendi'):
        e, f = dd(lam, k), dd(full, k)
        for m in smodes[1:]:
            b = dd(m, k)
            pt = (e.mean()-b.mean())/(e.mean()-f.mean())
            bs = []
            for _ in range(B):
                i = rng.integers(0, len(sp), len(sp))
                bs.append((e[i].mean()-b[i].mean())/(e[i].mean()-f[i].mean()))
            shares[(m, k)] = (pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    for m in smodes[1:]:
        out.append(f'| {SPLIT_LABEL[m.split("+")[1]]} | {fmt(shares[(m, "broken")], 2)} | {fmt(shares[(m, "vendi")], 2)} |')
    for k, n in (('broken', '망가짐'), ('vendi', 'ΔVendi')):
        s = shares[(f'{lam}+readk', k)][0]+shares[(f'{lam}+readv', k)][0]-1
        out.append(f'- {n}: K 몫 + V 몫 − 1 = {s:+.2f} (0이면 K와 V가 더해지는 대로, 0에서 멀면 상호작용)')
    out += ['', '**픽셀 변화** (같은 prompt·seed, 평균 |차| / 255, 처음 80장):', '']
    for m in smodes:
        v, n = pixdiff(model, lam, m)
        out.append(f'- {m} vs {lam}: {v:.1f} (n {n})')
    v, n = pixdiff(model, fixtxt, full)
    out.append(f'- {full} vs {fixtxt} (같은 조작, 보정항 fp32 대 bf16 재계산): {v:.1f} (n {n})')
    return out


if __name__ == '__main__':
    main()
