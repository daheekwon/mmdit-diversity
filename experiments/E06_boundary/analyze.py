"""E06 analysis (PROTOCOL_ko.md §4-§6, amendments §10-§13): per-condition table against Original, the E1 knee test, the
E2 onset, the window end, and the E3 points. Writes outputs/e06_boundary/<model>/e06/REPORT_<model>.md and prints the E3
decision.

  python experiments/E06_boundary/analyze.py sd35 [e06|ho4]

Definitions (PROTOCOL §4, §11): broken = p_defect >= .5 or p_style >= .5; mismatch = p_match < .5; rates in pp per prompt
(8 seeds), differences are paired by prompt; 95% intervals from 2,000 prompt-bootstrap resamples (seed 0). Judge rows
judged twice (evaluation passes split shards differently) are identical and are de-duplicated.
"""
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402

B = 2000
MARGIN = 2.0


DARK_LUMA, DARK_SHARE = 25, .3            # value_diversity/dark_fail.py: dark image = share of luma < 25 above 30%


def dark_table(root):
    """Per-image dark-pixel share (dark_fail.dark_share), cached in <root>/dark_share.csv."""
    from PIL import Image
    path, have = root/'dark_share.csv', {}
    if path.exists():
        have = {(r['pid'], r['mode'], r['seed']): float(r['dark']) for r in csv.DictReader(path.open())}
    gen, new = root/'generation', []
    for img in sorted(gen.glob('*/*/seed*.png')):
        k = (img.parent.parent.name, img.parent.name, img.stem)
        if k not in have:
            a = np.asarray(Image.open(img).convert('L').resize((128, 128)), dtype=float)
            have[k] = float((a < DARK_LUMA).mean())
            new.append(k)
    if new:
        with path.open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pid', 'mode', 'seed', 'dark'])
            for k, v in sorted(have.items()):
                w.writerow([*k, round(v, 5)])
    return have


def load(model, split='e06'):
    root = C.gen_root(model, split).parent
    J = {}
    for f in sorted(root.glob('judge_e06_*of2.csv')):
        for r in csv.DictReader(f.open()):
            J[(r['pid'], r['mode'], r['seed'])] = r
    per = {}                                        # (pid, mode) -> dict of lists

    def add(pid, mode, key, val):
        per.setdefault((pid, mode), {}).setdefault(key, []).append(val)
    for (p, m, s), r in J.items():
        d, st, ma = float(r['p_defect']) >= .5, float(r['p_style']) >= .5, float(r['p_match']) < .5
        add(p, m, 'broken', 100.*(d or st))
        add(p, m, 'mismatch', 100.*ma)
        add(p, m, 'defect', 100.*d)
        add(p, m, 'style', 100.*st)
    for r in csv.DictReader((root/'eval_combo_images.csv').open()):
        add(r['pid'], r['mode'], 'musiq', float(r['musiq']))
    dark = dark_table(root)
    for (p, m, s), v in dark.items():
        if (p, m, s) in J:
            r = J[(p, m, s)]
            add(p, m, 'dark', 100.*(v > DARK_SHARE))
            if v > DARK_SHARE:                       # mismatch among dark images (PROTOCOL §4, P7)
                add(p, m, 'dark_mis', 100.*(float(r['p_match']) < .5))
    if (root/'vqa_0of1.csv').exists():
        for r in csv.DictReader((root/'vqa_0of1.csv').open()):
            add(r['pid'], r['mode'], 'vqa', float(r['vqa_mean']))
    for r in csv.DictReader((root/'eval_combo_diversity.csv').open()):
        for k in ('comp', 'color'):
            add(r['pid'], r['mode'], k, float(r[k]))
    vrows = list(csv.DictReader((root/'eval_combo_vendi.csv').open()))
    vendi = {(r['pid'], r['mode']): float(r['vendi_dino']) for r in vrows}
    vclip = {(r['pid'], r['mode']): float(r['vendi_clip']) for r in vrows if r.get('vendi_clip')}   # auxiliary (11-06)
    T = {}
    for (p, m), d in per.items():
        T[(p, m)] = {k: float(np.mean(v)) for k, v in d.items()}
        if (p, m) in vendi:
            T[(p, m)]['vendi'] = vendi[(p, m)]
        if (p, m) in vclip:
            T[(p, m)]['vendi_clip'] = vclip[(p, m)]
    return T


def prompts_of(T, mode):
    return sorted(p for (p, m) in T if m == mode)


def delta(T, mode, key, ref='original', pids=None):
    ps = pids or sorted(set(prompts_of(T, mode)) & set(prompts_of(T, ref)))
    ps = [p for p in ps if key in T[(p, mode)] and key in T[(p, ref)]]     # VQA: rare / comp prompts only
    return np.array([T[(p, mode)][key]-T[(p, ref)][key] for p in ps]), ps


def boot_mean(d, rng):
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(t, nd=1):
    return f'{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]'


# ---------------------------------------------------------------- knee test (PROTOCOL §5, §10, §11)
def fit_line(x, y):
    a = float((x*y).sum()/(x*x).sum())
    return a, float(((y-a*x)**2).sum())


def fit_piece(x, y, bps):
    best = None
    for b in bps:
        X = np.stack([np.minimum(x, b), np.maximum(x-b, 0.)], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y-X@coef)**2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(coef[1]), float(b))
    return best


def ratio(a, c):
    """Back / front slope, as the protocol states it. A front slope <= 0 gives no ratio (the criterion fails); it is
    not read as an infinite ratio."""
    return c/a if a > 0 else -math.inf


def knee(T, modes, key, rng):
    """modes: edited conditions of one curve, in increasing strength. Points = prompt x condition."""
    pids = sorted(set.intersection(*[set(prompts_of(T, m)) for m in modes+['original']]))
    X = np.array([[T[(p, m)]['vendi']-T[(p, 'original')]['vendi'] for m in modes] for p in pids])
    Y = np.array([[T[(p, m)][key]-T[(p, 'original')][key] for m in modes] for p in pids])

    def run(X, Y):
        bps = X.mean(0)[1:-1]
        x, y = X.ravel(), Y.ravel()
        a, sse1 = fit_line(x, y)
        sse3, a2, c2, b = fit_piece(x, y, bps)
        n = len(x)
        aic1, aic3 = n*math.log(sse1/n)+2, n*math.log(sse3/n)+6
        return dict(slope=a, a=a2, c=c2, bp=b, dAIC=aic1-aic3, ratio=ratio(a2, c2), n=n)
    res = run(X, Y)
    rs, sl = [], []
    for _ in range(B):
        i = rng.integers(0, len(pids), len(pids))
        r = run(X[i], Y[i])
        rs.append(r['ratio'])
        sl.append(r['slope'])
    rs = np.sort(np.array(rs))                      # -inf (front slope <= 0) sorts first and counts as failing
    res['ratio_ci'] = (float(rs[int(.025*B)]), float(rs[int(.975*B)-1]))
    res['front_le0'] = float(np.mean(~np.isfinite(rs)))
    res['slope_ci'] = (float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5)))
    res['knee'] = bool(res['dAIC'] > 2 and math.isfinite(res['ratio']) and res['ratio'] >= 2 and res['ratio_ci'][0] > 1)
    bp_mode = min(modes[1:-1], key=lambda m: abs(np.mean([T[(p, m)]['vendi']-T[(p, 'original')]['vendi'] for p in pids])-res['bp']))
    res['bp_mode'] = bp_mode
    return res


def window_end(rows, key):
    """rows: [(mode, dVendi, dCost)] in increasing strength, starting at Original (0, 0). Linear interpolation of the
    first +2pp crossing; None if no crossing."""
    for (m0, x0, y0), (m1, x1, y1) in zip(rows, rows[1:]):
        if y1 > MARGIN >= y0:
            return x0+(MARGIN-y0)*(x1-x0)/(y1-y0), m1
        if y0 > MARGIN:
            return x0, m0
    return None, None


def main():
    model = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else 'e06'
    rng = np.random.default_rng(0)
    T = load(model, split)
    ref = 'original' if not (split == 'ho4' and model == 'flux') else 'b8_original'
    if ref != 'original':                              # FLUX ho4: the linked Original is b8_original
        T.update({(p, 'original'): v for (p, m), v in list(T.items()) if m == ref})
    allmodes = sorted({m for (_, m) in T})
    E1 = sorted([m for m in allmodes if m.startswith('v_l') and '+' not in m], key=lambda m: float(m[3:]))
    E2 = [f'padv{f}' for f in ('0.5', '0.25', '0.125', '0') if f'padv{f}' in allmodes]
    final = 'v_l1.14+padv0.25' if model == 'sd35' else 'v_l1.72+padv0.25'
    low = f'v_l{C.COMBO_LOW[model]}+padv0.25'
    rare = [p for p in prompts_of(T, 'original') if p.startswith('rare_')]
    out = [f'# E06 {model} {split} 결과 표 (원본 대비, prompt 짝 bootstrap 95%)', '']
    keys = [('vendi', 'ΔVendi', 2), ('broken', '망가짐', 1), ('mismatch', '불일치', 1), ('defect', 'defect', 1),
            ('style', 'style', 1), ('musiq', 'MUSIQ', 2)]
    out.append('| 조건 | n | '+' | '.join(k[1] for k in keys)+' | 창 (전체 / 희귀) |')
    out.append('|'+'---|'*(len(keys)+3))
    summary = {}
    for m in [x for x in allmodes if x not in ('original', ref)]:
        cells, ok, ok_rare = [], True, True
        for k, _, nd in keys:
            d, ps = delta(T, m, k)
            t = boot_mean(d, rng)
            cells.append(fmt(t, nd))
            summary.setdefault(m, {})[k] = t[0]
            dr, _ = delta(T, m, k, pids=[p for p in rare if p in ps])
            summary[m][k+'_rare'] = float(dr.mean()) if len(dr) else float('nan')
        s = summary[m]
        win = lambda sfx: ('밖' if s['broken'+sfx] > MARGIN or s['mismatch'+sfx] > MARGIN else
                           ('플래그' if s['musiq'+sfx] < -1 else '안'))
        out.append(f'| {m} | {len(ps)} | '+' | '.join(cells)+f' | {win("")} / {win("_rare")} |')
    out += ['', '- 창: 망가짐·불일치 − 원본 ≤ +2pp(점추정)면 안, MUSIQ − 원본 < −1이면 플래그(시트 규칙 대상). ΔVendi 구간 하한 > 0 조건은 표의 ΔVendi 구간으로 본다.', '']
    extra = [('vqa', 'VQA(희귀·구성)', 3), ('comp', '구도 다양성', 3), ('color', '색 다양성', 1), ('dark', '어두운 비율(기술용)', 1)]
    out += ['## 보조 지표 (원본 대비)', '', '| 조건 | '+' | '.join(e[1] for e in extra)+' | 원본 값 |', '|'+'---|'*(len(extra)+2)]
    for m in [x for x in allmodes if x not in ('original', ref)]:
        cells = []
        for k, _, nd in extra:
            try:
                d, _ = delta(T, m, k)
                cells.append(fmt(boot_mean(d, rng), nd) if len(d) else '—')
            except KeyError:
                cells.append('—')
        base = {k: np.mean([T[(p, 'original')][k] for p in prompts_of(T, 'original') if k in T[(p, 'original')]] or [np.nan]) for k, _, _ in extra}
        out.append(f'| {m} | '+' | '.join(cells)+' | '+', '.join(f'{k} {v:.2f}' for k, v in base.items())+' |')
    dm = {m: [T[(p, m)]['dark_mis'] for p in prompts_of(T, m) if 'dark_mis' in T[(p, m)]] for m in allmodes}
    out += ['', '- 어두운 이미지 안의 불일치율(이미지 수 가중 아님, prompt 평균): '+', '.join(
        f'{m} {np.mean(v):.1f}% (prompt {len(v)})' for m, v in dm.items() if v), '']
    if split != 'e06':
        rep = C.gen_root(model, split).parent/f'REPORT_{model}_{split}.md'
        rep.write_text('\n'.join(out)+'\n')
        print('\n'.join(out))
        return
    # knee
    out += ['## E1 무릎 판정 (이미지 V만)', '', f'곡선: {", ".join(E1)}', '']
    knees = {}
    for key, name in (('broken', '망가짐'), ('mismatch', '불일치')):
        r = knee(T, E1, key, rng)
        knees[key] = r
        out.append(f'- {name}: 직선 기울기 {r["slope"]:+.2f} pp/Vendi [{r["slope_ci"][0]:+.2f}, {r["slope_ci"][1]:+.2f}], '
                   f'꺾은선 앞 {r["a"]:+.2f} / 뒤 {r["c"]:+.2f} (꺾는 점 ΔVendi {r["bp"]:.2f}, 가까운 조건 {r["bp_mode"]}), '
                   f'ΔAIC {r["dAIC"]:.1f}, 기울기 비 {r["ratio"]:.2f} [{r["ratio_ci"][0]:.2f}, {r["ratio_ci"][1]:.2f}] (앞 기울기 ≤ 0인 bootstrap {100*r["front_le0"]:.0f}%), n {r["n"]} → '
                   f'**{"무릎 있음" if r["knee"] else "무릎 없음"}**')
    # window ends
    out += ['', '## 창이 끝나는 ΔVendi (점추정 곡선, 선형 보간)', '']
    curves = dict(E1=E1, E2=E2, 조합=['padv0.25', low, final])
    for name, ms in curves.items():
        for key, kn in (('broken', '망가짐'), ('mismatch', '불일치')):
            rows = [('original', 0., 0.)]+[(m, summary[m]['vendi'], summary[m][key]) for m in ms if m in summary]
            x, m = window_end(rows, key)
            out.append(f'- {name} {kn}: ' + ('범위 끝까지 창 안' if x is None else f'ΔVendi {x:.2f}에서 +2pp를 넘음 (넘는 첫 조건 {m})'))
    # E2 onset
    onset = next((m for m in E2 if summary[m]['broken'] > MARGIN or summary[m]['mismatch'] > MARGIN), 'padv0')
    # E3 points
    kb = knees['broken']
    if kb['knee'] or knees['mismatch']['knee']:
        k = kb if kb['knee'] else knees['mismatch']
        i = E1.index(k['bp_mode'])
        vpts = [E1[i], E1[min(i+1, len(E1)-1)]]
        why = f'무릎 있음({k["bp_mode"]}) → 무릎 전후'
    else:
        vpts = C.E3_V[model]
        why = '무릎 없음 → §3 기본 점'
    out += ['', '## E3 점 (PROTOCOL §3 규칙)', '', f'- 이미지 V: {", ".join(vpts)} ({why})',
            f'- pad: {onset} (E2에서 망가짐 또는 불일치 − 원본이 처음 +2pp를 넘는 s; 없으면 padv0)']
    rep = C.gen_root(model, 'e06').parent/f'REPORT_{model}.md'
    rep.write_text('\n'.join(out)+'\n')
    print('\n'.join(out))
    print(json.dumps(dict(model=model, e3_v=vpts, e3_pad=onset, knee={k: v['knee'] for k, v in knees.items()})))


if __name__ == '__main__':
    main()
