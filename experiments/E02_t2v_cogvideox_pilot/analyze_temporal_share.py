"""Tables of PROTOCOL_ko.md section 7b from outputs/cogvideox/pilot/temporal_share.json (see temporal_share.py).

  python experiments/E02_t2v_cogvideox_pilot/analyze_temporal_share.py

Shares are energy-weighted (sum of temporal-variation energy / sum of energy) unless stated. Prompt-level bootstrap 95%
intervals. The flicker increase per prompt is ours - original (and V edit only - original) of the mean adjacent-frame
pixel difference (metrics/per_video.csv of the stage-2a runs). Writes outputs/cogvideox/pilot/temporal_share.md.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from div import config                                                        # noqa: E402
from div.analysis import bootstrap                                            # noqa: E402

PILOT = config.ROOT/'outputs'/'cogvideox'/'pilot'
RUNS = config.ROOT/'outputs'/'cogvideox'/'vbench_tune'


def shares(g):
    s = g[['edit_all', 'edit_tv', 'x_all', 'x_tv', 'dq_all', 'dq_tv', 'd_all', 'd_tv']].sum()
    return pd.Series(dict(
        edit=s.edit_tv/s.edit_all,                                            # added value Q Q^T x
        outside_x=(s.x_tv-s.edit_tv)/(s.x_all-s.edit_all),                    # x outside Q
        dev_in_q=s.dq_tv/s.dq_all,                                            # seed deviation in Q
        dev_out_q=(s.d_tv-s.dq_tv)/(s.d_all-s.dq_all),                        # seed deviation outside Q
        q_share_of_x=s.edit_all/s.x_all, q_share_of_dev=s.dq_all/s.d_all))


def main():
    import json
    raw = json.loads((PILOT/'temporal_share.json').read_text())
    df = pd.concat([pd.DataFrame(rows).assign(id=pid) for pid, rows in raw.items()], ignore_index=True)
    lines = [f'prompts {df.id.nunique()}, rows {len(df)} (step x block x branch per prompt {len(df)//df.id.nunique()})', '']
    per = df.groupby('id').apply(shares)
    allv = shares(df)
    lines += ['## pooled (energy-weighted) with prompt bootstrap 95% interval of the per-prompt shares', '',
              '| share | pooled | prompt mean | 95% interval |', '|---|---|---|---|']
    for c in allv.index:
        lo, hi = bootstrap(per[c])
        lines.append(f'| {c} | {allv[c]:.3f} | {per[c].mean():.3f} | [{lo:.3f}, {hi:.3f}] |')
    lines += ['', 'reference: frames independent 12/13 = 0.923, identical frames 0', '']
    for key, title in (('step', 'by step'), ('block', 'by block'), ('branch', 'by CFG branch')):
        t = df.groupby(key).apply(shares)[['edit', 'outside_x', 'dev_in_q', 'dev_out_q', 'q_share_of_x']]
        lines += [f'## {title}', '', t.round(3).to_markdown(), '']
    fl = {t: pd.read_csv(RUNS/t/'metrics'/'per_video.csv').groupby('id').flicker_mae.mean() for t in ('original', 'ours', 'v_edit')}
    per['flicker_ours'] = fl['ours']-fl['original']
    per['flicker_v_edit'] = fl['v_edit']-fl['original']
    lines += ['## per prompt', '', per.round(3).to_markdown(), '']
    for c in ('flicker_ours', 'flicker_v_edit'):
        r = spearmanr(per['edit'], per[c])
        lines.append(f'Spearman(edit share, {c}) = {r.statistic:.3f} (p {r.pvalue:.3f}, 16 prompts, descriptive)')
    text = '\n'.join(lines)
    (PILOT/'temporal_share.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
