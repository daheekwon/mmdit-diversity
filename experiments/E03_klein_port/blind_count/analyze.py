"""Unblind counts.csv with key.json; per-condition eye-counted failures, judge overlap, Vendi without failures (CRITERIA_ko.md)."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, '/DATA0/dahee/diversity_migration')
from value_diversity.vendi_eval import vendi
H = Path(__file__).parent
EV = Path('/DATA0/dahee/diversity_migration/mmdit_diversity/outputs/klein/eval/ho3')
G, CACHE = EV/'generation', EV/'eval_cache'
key = json.load(open(H/'key.json'))
MODES = ['original', 'dave_a0.2~0-10'] + [f'q_l{l}@0-24~0-10+pad_v0.25~0-10' for l in (1.3, 1.5, 1.7)]
LAB = {'original': 'Original', 'dave_a0.2~0-10': 'DAVE 0-10', **{f'q_l{l}@0-24~0-10+pad_v0.25~0-10': f'ours l{l}' for l in (1.3, 1.5, 1.7)}}
STRUCT = ['panel', 'bar', 'insert', 'noise', 'deform']
c = pd.read_csv(H/'counts.csv')
c['mode'] = [key[p][r] for p, r in zip(c.pid, c.row)]
c['seed'] = 'seed' + (66000 + c.seed).astype(str)
c['src'] = 'eye'
import sys as _s
if '--final' in _s.argv:
    v = pd.read_csv(H/'judge_only_verified.csv'); v = v[v.real == 1].copy()
    v['seed'] = 'seed' + (66000 + v.seed).astype(str); v['src'] = 'judge_verified'
    c = pd.concat([c, v[['pid', 'mode', 'seed', 'kind', 'note', 'src']]], ignore_index=True)
pids = sorted(key)
N = len(pids)*16
# judge
j = pd.read_csv(EV/'judge_klein_0of1.csv').drop_duplicates(['pid', 'mode', 'seed'])
for k in ('p_match', 'p_defect', 'p_style'):
    j[k] = pd.to_numeric(j[k])
j['jbrk'] = (j.p_defect >= .5) | (j.p_style >= .5)
j = j[j['mode'].isin(MODES)].set_index(['pid', 'mode', 'seed'])
fail = set(zip(c.pid, c['mode'], c.seed))
struct = set(zip(*[c[c.kind.isin(STRUCT)][k] for k in ('pid', 'mode', 'seed')]))
rows = []
for m in MODES:
    cm = c[c['mode'] == m]
    jm = j.xs(m, level='mode')
    jb = set((p, s) for (p, s), v in jm.jbrk.items() if v)
    eye = set(zip(cm.pid, cm.seed))
    r = dict(cond=LAB[m], **{k: int((cm.kind == k).sum()) for k in STRUCT + ['style']})
    r['struct'] = sum(r[k] for k in STRUCT); r['all'] = len(cm)
    r['all_%'] = 100*r['all']/N; r['struct_%'] = 100*r['struct']/N
    r['judge_brk'] = len(jb); r['eye&judge'] = len(eye & jb); r['eye_only(judge missed)'] = len(eye - jb); r['judge_only'] = len(jb - eye)
    r['prompts_with_fail'] = cm.pid.nunique()
    rows.append(r)
T = pd.DataFrame(rows).set_index('cond')
# seed concentration: how many failures fall on seed 5 (index) per condition
T['on_seed5'] = [int(((c['mode'] == m) & (c.seed == 'seed66005')).sum()) for m in MODES]
T['on_seed1'] = [int(((c['mode'] == m) & (c.seed == 'seed66001')).sum()) for m in MODES]

# Vendi without failures (matched k per prompt)
def feats(pid, m):
    out = {}
    for q in sorted((G/pid/m).glob('seed*.png')):
        cp = CACHE/f"{hashlib.sha256(q.read_bytes()+pid.encode()+b'v1').hexdigest()}.npz"
        with np.load(cp) as z:
            out[q.stem] = z['dino']
    return out
rng = np.random.default_rng(0)
V, skipped, ks = {}, [], {}
for pid in pids:
    F = {m: feats(pid, m) for m in MODES}
    valid = {m: [s for s in F[m] if (pid, m, s) not in fail] for m in MODES}
    k = min(len(v) for v in valid.values())
    ks[pid] = k
    if k < 6:
        skipped.append(pid); continue
    V[pid] = {}
    for m in MODES:
        allS = list(F[m])
        def ev(S):
            return float(np.mean([vendi(np.stack([F[m][s] for s in rng.choice(S, k, replace=False)])) for _ in range(500)]))
        V[pid][m] = dict(clean=ev(valid[m]), all_k=ev(allS), full=vendi(np.stack([F[m][s] for s in allS])))
P = sorted(V)
def stat(field):
    out = {}
    for m in MODES:
        d = np.array([V[p][m][field] - V[p]['original'][field] for p in P])
        bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
        out[LAB[m]] = (d.mean(), *np.percentile(bs, [2.5, 97.5]))
    return out
def vs_dave(field):
    out = {}
    for m in MODES:
        d = np.array([V[p][m][field] - V[p]['dave_a0.2~0-10'][field] for p in P])
        bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
        out[LAB[m]] = (d.mean(), *np.percentile(bs, [2.5, 97.5]))
    return out
res = dict(table=T.reset_index().to_dict('records'), k=ks, skipped=skipped, n_prompts_vendi=len(P),
           dV_full=stat('full'), dV_all_k=stat('all_k'), dV_clean=stat('clean'), vsDAVE_clean=vs_dave('clean'), vsDAVE_all_k=vs_dave('all_k'),
           per_prompt_fail={LAB[m]: c[c['mode'] == m].groupby('pid').size().to_dict() for m in MODES})
json.dump(res, open(H/('result_final.json' if '--final' in _s.argv else 'result.json'), 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print(T.to_string())
print('\nk per prompt:', ks, '\nskipped (k<6):', skipped, ' prompts used:', len(P))
for f in ('dV_full', 'dV_all_k', 'dV_clean', 'vsDAVE_all_k', 'vsDAVE_clean'):
    print(f'\n{f}:'); [print(f'  {n:10s} {a:+.2f} [{b:+.2f}, {cc:+.2f}]') for n, (a, b, cc) in res[f].items()]
print('\nper prompt failures:'); print(pd.DataFrame(res['per_prompt_fail']).fillna(0).astype(int).to_string())
