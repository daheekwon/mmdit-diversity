"""Verdict tables of PROTOCOL_ko.md (sections 5 and 6) -> REPORT_tables.md next to this file.

Per model and resolution, from outputs/res<r>/<model>/coco48/: eval_combo_vendi.csv, eval_combo_images.csv and
judge_combo_R_*.csv. broken = p_defect >= .5 or p_style >= .5; mismatch = p_match < .5 (as in the 512 COCO check).
Paired bootstrap over the 48 prompts (10,000 draws, the same draws at both resolutions).

  python experiments/E01_res1024_coco48/analyze.py [--models sd35 flux]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]/'outputs'
MODES = dict(sd35=dict(orig='original', dave='dave_a0.5', ours='q_l1.4+padv0.125'),
             flux=dict(orig='b8_original', dave='dave_a0.2', ours='b8_q_d0.7+pad_v0.25'))
NAMES = dict(orig='Original', dave='DAVE', ours='ours')
RES = (512, 1024)
B = 10000


def per_prompt(model, res):
    """DataFrame indexed by (pid, role) with vendi_dino, vendi_clip, broken, mismatch, style, musiq, clip_text."""
    base = OUT/f'res{res}'/model/'coco48'
    roles = {v: k for k, v in MODES[model].items()}
    v = pd.read_csv(base/'eval_combo_vendi.csv')
    im = pd.read_csv(base/'eval_combo_images.csv')
    j = pd.concat([pd.read_csv(p) for p in sorted(base.glob('judge_combo_R_*.csv'))])
    j = j.assign(broken=((j.p_defect >= .5) | (j.p_style >= .5)).astype(float), mismatch=(j.p_match < .5).astype(float),
                 style=(j.p_style >= .5).astype(float))
    parts = [v.set_index(['pid', 'mode'])[['n', 'vendi_dino', 'vendi_clip']],
             j.groupby(['pid', 'mode'])[['broken', 'mismatch', 'style']].mean()*100,
             im.groupby(['pid', 'mode'])[['musiq', 'clip_text']].mean()]
    df = pd.concat(parts, axis=1).reset_index()
    df = df[df['mode'].isin(roles)].assign(role=lambda d: d['mode'].map(roles))
    if not (df['n'] == 8).all():
        raise ValueError(f'{model} {res}: not every (prompt, mode) has 8 images')
    return df.set_index(['pid', 'role']).drop(columns=['mode', 'n']).unstack('role')


def boot(x, idx):
    """Mean and 95% interval of the per-prompt values x [P] over bootstrap draws idx [B, P]."""
    m = x[idx].mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def fmt(t, d=3):
    return f'{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]'


def model_report(model, rng):
    tabs = {r: per_prompt(model, r) for r in RES}
    pids = sorted(set(tabs[512].index) & set(tabs[1024].index))
    if len(pids) != 48:
        raise ValueError(f'{model}: {len(pids)} prompts in both resolutions, expected 48')
    tabs = {r: t.loc[pids] for r, t in tabs.items()}
    idx = rng.integers(0, len(pids), (B, len(pids)))
    lines = [f'## {model}', '', '### Mean per condition (8 seeds per prompt, 48 prompts)', '',
             '| res | condition | Vendi DINO | Vendi CLIP | broken % | mismatch % | style % | MUSIQ | CLIP-text |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in RES:
        for role in ('orig', 'dave', 'ours'):
            t = tabs[r]
            lines.append(f'| {r} | {NAMES[role]} | ' + ' | '.join(
                f'{t[(c, role)].mean():.3f}' if c.startswith('vendi') or c == 'clip_text' else f'{t[(c, role)].mean():.2f}'
                for c in ('vendi_dino', 'vendi_clip', 'broken', 'mismatch', 'style', 'musiq', 'clip_text')) + ' |')
    d = {r: {role: (tabs[r][('vendi_dino', role)]-tabs[r][('vendi_dino', 'orig')]).to_numpy() for role in ('dave', 'ours')}
         for r in RES}
    res = {}
    # rule 1
    r1 = boot(d[1024]['ours'], idx)
    res['rule1'] = r1[1] > 0
    # rule 2
    num, den = d[1024]['ours'][idx].mean(1), d[512]['ours'][idx].mean(1)
    ratio = (d[1024]['ours'].mean()/d[512]['ours'].mean(), *np.percentile(num/den, [2.5, 97.5]))
    res['rule2'] = ratio[0] >= .5
    lines += ['', '### Rules 1-2: our ΔVendi(DINO) vs Original', '',
              '| | 512 | 1024 | 1024 ÷ 512 |', '|---|---|---|---|',
              f"| ours | {fmt(boot(d[512]['ours'], idx))} | {fmt(r1)} | {ratio[0]:.2f} [{ratio[1]:.2f}, {ratio[2]:.2f}] |",
              f"| DAVE (descriptive) | {fmt(boot(d[512]['dave'], idx))} | {fmt(boot(d[1024]['dave'], idx))} | "
              f"{d[1024]['dave'].mean()/d[512]['dave'].mean():.2f} |",
              f"| Original Vendi, 1024 − 512 (descriptive) | | {fmt(boot((tabs[1024][('vendi_dino', 'orig')]-tabs[512][('vendi_dino', 'orig')]).to_numpy(), idx))} | |",
              '', f"Rule 1 (1024 lower bound > 0): **{'pass' if res['rule1'] else 'fail'}**. "
              f"Rule 2 (ratio ≥ 0.5): **{'pass' if res['rule2'] else 'fail'}**."]
    # rule 3
    lines += ['', '### Rule 3: ours − DAVE at each resolution', '',
              '| res | ΔVendi diff | broken pp | mismatch pp | MUSIQ | diversity verdict (≥ 0) | cost verdict (≤ +2, ≤ +2, ≥ −1) |',
              '|---|---|---|---|---|---|---|']
    verdicts = {}
    for r in RES:
        t = tabs[r]
        diff = {c: (t[(c, 'ours')]-t[(c, 'dave')]).to_numpy() for c in ('vendi_dino', 'broken', 'mismatch', 'musiq')}
        b = {c: boot(x, idx) for c, x in diff.items()}
        div_ok = b['vendi_dino'][0] >= 0
        cost_ok = b['broken'][0] <= 2 and b['mismatch'][0] <= 2 and b['musiq'][0] >= -1
        verdicts[r] = (div_ok, cost_ok)
        lines.append(f"| {r} | {fmt(b['vendi_dino'])} | {fmt(b['broken'], 1)} | {fmt(b['mismatch'], 1)} | {fmt(b['musiq'], 2)} | "
                     f"{'pass' if div_ok else 'fail'} | {'pass' if cost_ok else 'fail'} |")
    res['rule3'] = verdicts[512] == verdicts[1024]
    lines += ['', f"Rule 3 (both verdicts the same at 512 and 1024): **{'kept' if res['rule3'] else 'changed'}**."]
    return lines, res


def mech_report(model):
    paths = {r: OUT/f'res{r}'/model/'coco48'/'mech.json' for r in RES}
    if not all(p.exists() for p in paths.values()):
        return [f'### Mechanism ({model}): not measured yet']
    m = {r: json.loads(p.read_text()) for r, p in paths.items()}
    lines = [f'### Mechanism ({model}, Original, prompts {m[512]["pids"][0]}–{m[512]["pids"][-1]}, 2 seeds)', '',
             '| res | pad mass cond (steps 0–2, all blocks) | share of image queries with pad mass > 0.5 | '
             'pad mass uncond | V change at our operating point | λ |', '|---|---|---|---|---|---|']
    for r in RES:
        cells = [c for p in m[r]['pad_mass'].values() for c in p.values()]
        vc = [x for p in m[r]['v_change'].values() for x in p.values()]
        unc = [c['uncond']['mean'] for c in cells if 'uncond' in c]
        lines.append(f"| {r} | {np.mean([c['cond']['mean'] for c in cells]):.4f} | "
                     f"{np.mean([c['cond']['share_gt_half'] for c in cells]):.4f} | "
                     f"{np.mean(unc) if unc else float('nan'):.4f} | {np.mean(vc):.4f} | {m[r]['lam']:.4f} |")
    by_step = {r: {} for r in RES}
    for r in RES:
        for p in m[r]['pad_mass'].values():
            for key, c in p.items():
                by_step[r].setdefault(int(key.split('/')[0]), []).append(c['cond']['mean'])
    lines += ['', 'Pad mass (cond) by step: ' + '; '.join(
        f"step {s}: {np.mean(by_step[512][s]):.4f} → {np.mean(by_step[1024][s]):.4f}" for s in sorted(by_step[512]))]
    return lines


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--models', nargs='*', default=['sd35', 'flux'])
    x = a.parse_args()
    rng = np.random.default_rng(0)
    lines, summary = ['# E01 1024 check: tables (generated by analyze.py)', ''], {}
    for model in x.models:
        body, summary[model] = model_report(model, rng)
        lines += body + [''] + mech_report(model) + ['']
    lines += ['## Summary', '', '| model | rule 1 | rule 2 | rule 3 |', '|---|---|---|---|']
    lines += [f"| {m} | {'pass' if s['rule1'] else 'fail'} | {'pass' if s['rule2'] else 'fail'} | "
              f"{'kept' if s['rule3'] else 'changed'} |" for m, s in summary.items()]
    (HERE/'REPORT_tables.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
