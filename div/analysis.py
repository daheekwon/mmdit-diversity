"""Comparisons between runs of the same model and dataset.

compare     per-prompt metrics: each run's mean and its difference from a reference run, with a 95% bootstrap interval
            over prompts (paired: the same prompts and seeds in every run); run-level metrics side by side.
            Main use: our method at its own operating point against DAVE's standard setting, Vendi and costs side by side.
interpolate a cost at a common diversity level along a strength sweep, for comparing our variants with each other
            (several levels in the common range, or the operating point's Vendi; never fitted to DAVE's Vendi).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load(run):
    run = Path(run)
    per = pd.read_csv(run/'metrics'/'per_prompt.csv', index_col='id')
    return per, json.loads((run/'metrics'/'summary.json').read_text())


def bootstrap(d, n=10000, seed=0):
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    return np.percentile(d[idx].mean(1), [2.5, 97.5])


def compare(ref, runs, metrics=None):
    """Table: one row per (run, metric) with mean, difference to `ref` and its bootstrap interval."""
    base, base_sum = load(ref)
    rows = []
    for run in runs:
        per, summ = load(run)
        common = base.index.intersection(per.index)
        for m in metrics or [c for c in per.columns if c in base.columns]:
            d = per.loc[common, m]-base.loc[common, m]
            lo, hi = bootstrap(d)
            rows.append(dict(run=Path(run).name, metric=m, mean=per.loc[common, m].mean(), ref=base.loc[common, m].mean(),
                             diff=d.mean(), lo=lo, hi=hi, prompts=len(common)))
        for m in ('fid', 'precision', 'recall', 'density', 'coverage'):
            if m in summ:
                rows.append(dict(run=Path(run).name, metric=m, mean=summ[m], ref=base_sum.get(m),
                                 diff=summ[m]-base_sum[m] if m in base_sum else None))
    return pd.DataFrame(rows)


def interpolate(x, y, x0):
    """Linear interpolation of y at x0 along a sweep sorted by x; None when x0 is outside the sweep."""
    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    if not x[0] <= x0 <= x[-1]:
        return None
    return float(np.interp(x0, x, y))
