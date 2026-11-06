"""Verdict tables of PROTOCOL_ko.md section 6 -> REPORT_tables.md next to this file.

Vendi and per-image scores: outputs/e04_final/res<r>/<model>/coco48/eval_combo_{vendi,images}.csv (all three conditions).
Judge: the final method from the E04 folder, Original and DAVE from E01's judge_combo_R CSV (same judge, same images).
broken = p_defect >= .5 or p_style >= .5; mismatch = p_match < .5. Paired bootstrap over the 48 prompts (10,000 draws,
the same draws at both resolutions).

  python experiments/E04_res1024_transfer/analyze.py [--models sd35 flux]
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAVE, FINAL, ORIG, E01, OUT                                # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = dict(orig='Original', dave='DAVE', final='final')
RES = (512, 1024)
B = 10000


def judge(base, modes):
    j = pd.concat([pd.read_csv(p) for p in sorted(base.glob('judge_combo_R_*.csv'))])
    return j[j['mode'].isin(modes)]


def per_prompt(model, res):
    """DataFrame indexed by pid, columns (metric, role)."""
    roles = {ORIG[model]: 'orig', DAVE[model]: 'dave', FINAL[model]: 'final'}
    base, e01 = OUT/f'res{res}'/model/'coco48', E01/f'res{res}'/model/'coco48'
    v = pd.read_csv(base/'eval_combo_vendi.csv')
    im = pd.read_csv(base/'eval_combo_images.csv')
    j = pd.concat([judge(e01, [ORIG[model], DAVE[model]]), judge(base, [FINAL[model]])])
    j = j.assign(broken=((j.p_defect >= .5) | (j.p_style >= .5)).astype(float), mismatch=(j.p_match < .5).astype(float),
                 style=(j.p_style >= .5).astype(float))
    if (j.groupby(['pid', 'mode']).size() != 8).any():
        raise ValueError(f'{model} {res}: judge rows are not 8 per (prompt, mode)')
    parts = [v.set_index(['pid', 'mode'])[['n', 'vendi_dino', 'vendi_clip']],
             j.groupby(['pid', 'mode'])[['broken', 'mismatch', 'style']].mean()*100,
             im.groupby(['pid', 'mode'])[['musiq', 'clip_text']].mean()]
    df = pd.concat(parts, axis=1).reset_index()
    df = df[df['mode'].isin(roles)].assign(role=lambda d: d['mode'].map(roles))
    if len(df) != 48*3 or not (df['n'] == 8).all():
        raise ValueError(f'{model} {res}: expected 48 prompts x 3 conditions x 8 images, got {len(df)} rows')
    return df.set_index(['pid', 'role']).drop(columns=['mode', 'n']).unstack('role')


def boot(x, idx):
    m = x[idx].mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def fmt(t, d=3):
    return f'{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]'


def model_report(model, rng):
    tabs = {r: per_prompt(model, r) for r in RES}
    pids = sorted(tabs[512].index)
    tabs = {r: t.loc[pids] for r, t in tabs.items()}
    idx = rng.integers(0, len(pids), (B, len(pids)))
    lines = [f'## {model} (final: {FINAL[model]})', '', '### Mean per condition (8 seeds per prompt, 48 prompts)', '',
             '| res | condition | Vendi DINO | Vendi CLIP | broken % | mismatch % | style % | MUSIQ | CLIP-text |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in RES:
        for role in ('orig', 'dave', 'final'):
            t = tabs[r]
            lines.append(f'| {r} | {NAMES[role]} | ' + ' | '.join(
                f'{t[(c, role)].mean():.3f}' if c.startswith('vendi') or c == 'clip_text' else f'{t[(c, role)].mean():.2f}'
                for c in ('vendi_dino', 'vendi_clip', 'broken', 'mismatch', 'style', 'musiq', 'clip_text')) + ' |')
    out = {}
    # verdict 1: final - DAVE at each resolution
    lines += ['', '### Verdict 1: final − DAVE at each resolution (ΔVendi vs Original of the same resolution)', '',
              '| res | ΔVendi final | ΔVendi DAVE | ΔVendi diff | broken pp | mismatch pp | MUSIQ | verdict |',
              '|---|---|---|---|---|---|---|---|']
    for r in RES:
        t = tabs[r]
        dv = {role: (t[('vendi_dino', role)]-t[('vendi_dino', 'orig')]).to_numpy() for role in ('dave', 'final')}
        diff = {c: (t[(c, 'final')]-t[(c, 'dave')]).to_numpy() for c in ('vendi_dino', 'broken', 'mismatch', 'musiq')}
        b = {c: boot(x, idx) for c, x in diff.items()}
        ok = b['vendi_dino'][0] >= 0 and b['broken'][0] <= 2 and b['mismatch'][0] <= 2 and b['musiq'][0] >= -1
        out[f'v1_{r}'] = ok
        lines.append(f"| {r} | {fmt(boot(dv['final'], idx))} | {fmt(boot(dv['dave'], idx))} | {fmt(b['vendi_dino'])} | "
                     f"{fmt(b['broken'], 1)} | {fmt(b['mismatch'], 1)} | {fmt(b['musiq'], 2)} | {'pass' if ok else 'fail'} |")
    # verdict 2: 1024 vs 512, each vs its own Original
    d = {r: {c: (tabs[r][(c, 'final')]-tabs[r][(c, 'orig')]).to_numpy() for c in ('vendi_dino', 'broken', 'mismatch', 'musiq', 'style')}
         for r in RES}
    num, den = d[1024]['vendi_dino'][idx].mean(1), d[512]['vendi_dino'][idx].mean(1)
    ratio = (d[1024]['vendi_dino'].mean()/d[512]['vendi_dino'].mean(), *np.percentile(num/den, [2.5, 97.5]))
    inc = {c: boot(d[1024][c]-d[512][c], idx) for c in ('broken', 'mismatch', 'musiq', 'style')}
    ok_ratio = .67 <= ratio[0] <= 1.5
    ok_cost = inc['broken'][0] <= 2 and inc['mismatch'][0] <= 2 and inc['musiq'][0] >= -1
    out['v2'] = ok_ratio and ok_cost
    lines += ['', '### Verdict 2: final vs Original, 1024 compared with 512', '',
              '| | 512 | 1024 | 1024 vs 512 |', '|---|---|---|---|',
              f"| ΔVendi DINO | {fmt(boot(d[512]['vendi_dino'], idx))} | {fmt(boot(d[1024]['vendi_dino'], idx))} | "
              f"ratio {ratio[0]:.2f} [{ratio[1]:.2f}, {ratio[2]:.2f}] |"]
    for c, dd in (('broken', 1), ('mismatch', 1), ('style', 1), ('musiq', 2)):
        lines.append(f"| {c} | {fmt(boot(d[512][c], idx), dd)} | {fmt(boot(d[1024][c], idx), dd)} | "
                     f"difference {fmt(inc[c], dd)} |")
    lines += ['', f"Ratio in [0.67, 1.5]: **{'yes' if ok_ratio else 'no'}**. Cost (broken, mismatch ≤ +2 pp more than "
              f"512; MUSIQ not > 1 worse): **{'yes' if ok_cost else 'no'}**. Verdict 2: **{'pass' if out['v2'] else 'fail'}**.",
              '', 'Final-method broken % by prompt at 1024 (top 8): ' + ', '.join(
                  f'{p} {v:.0f}' for p, v in tabs[1024][('broken', 'final')].sort_values(ascending=False).head(8).items())]
    return lines, out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--models', nargs='*', default=['sd35', 'flux'])
    x = a.parse_args()
    rng = np.random.default_rng(0)
    lines, summary = ['# E04 final method at 512 and 1024: tables (generated by analyze.py)', ''], {}
    for model in x.models:
        body, summary[model] = model_report(model, rng)
        lines += body + ['']
    lines += ['## Summary', '', '| model | verdict 1 at 512 | verdict 1 at 1024 | verdict 2 |', '|---|---|---|---|']
    lines += [f"| {m} | {'pass' if s['v1_512'] else 'fail'} | {'pass' if s['v1_1024'] else 'fail'} | "
              f"{'pass' if s['v2'] else 'fail'} |" for m, s in summary.items()]
    (HERE/'REPORT_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
