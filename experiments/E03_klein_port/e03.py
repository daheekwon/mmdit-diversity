"""E03: our method on FLUX.2-klein-base-4B (PROTOCOL_ko.md). Job lists, evaluation layout, report.

  python experiments/E03_klein_port/e03.py jobs      # jobs_basis.txt, jobs_{gen,dave,eval}_<split>.txt (lanes.py format)
  python experiments/E03_klein_port/e03.py link      # outputs/klein/eval/<split>/generation/<pid>/<mode>/seedXXXXX.png
  python experiments/E03_klein_port/e03.py report    # REPORT_tables.md + verdict.json
  python experiments/E03_klein_port/e03.py sheets    # one prompt per figure, rows = conditions

The evaluation layout is the value_diversity/ one (symlinks + meta.json), so study_eval.py, vendi_eval.py and
study_judge.py run unchanged and the numbers are computed exactly as for SD3.5 / FLUX.1 in results/generalize/.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import sys
import numpy as np

E = Path(__file__).resolve().parent
ROOT = E.parents[1]                                   # mmdit_diversity
MIG = ROOT.parent                                     # diversity_migration
sys.path.insert(0, str(ROOT))
PY = '/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python'
OUT = ROOT/'outputs'/'klein'
EVAL = OUT/'eval'
SPLITS = ['dev', 'ho3', 'coco48']
LAMS = [1.3, 1.4, 1.5, 1.6, 1.7]
BLOCKS = ['5-24', '10-15', '0-24']                  # 0-24 = all blocks (double + single): added 11-05 at user request
STEPS = ['0-4', '0-6']
PAD = 0.25
DAVE = 'dave_a0.2'                                   # steps 0-19 (config); dev only since 11-05 20:00 (dropped from ho3)
# 11-05 20:00 user: ours and DAVE both on steps 0-10, ours on all blocks 0-24 at lam 1.3 / 1.5 / 1.7 (pad V 1/4 on the
# same steps), dev + ho3, run before the rest of the ho3 grid. DAVE steps 0-10 is the DAVE that rules 1-2 compare with
# wherever it exists (report: ref_dave).
LONG = '0-10'
LONG_LAMS = [1.3, 1.5, 1.7]
LONG_SPLITS = ['dev', 'ho3']
DAVE10 = f'dave_a0.2~{LONG}'
APPROVAL = E/'DAVE_APPROVED'
NEED_GEN = 16.                                        # GiB free on two checks (parity_klein: batch 16 peak 10.8 GiB)
NEED_BASIS = 20.
NEED_EVAL, NEED_JUDGE = 10., 24.
COCO_SHARDS = 1                                       # >1: shards of one run folder start together and race on prompts.jsonl
NAME = 'klein'                                        # eval / judge file name: eval_klein_*.csv, judge_klein_0of1.csv


def ours(lam, blocks, steps):
    return f'q_l{lam}@{blocks}~{steps}+pad_v{PAD}~{steps}'


def grid():
    return [(lam, b, s) for s in STEPS for b in BLOCKS for lam in LAMS]


def parts():
    """dev only (11-05 user request): each part of ours alone -> (mode, preset, overrides)."""
    return ([(f'pad_v{PAD}~{s}', 'pad_v', [f'pad_v.steps={s}', f'pad_v.factor={PAD}']) for s in STEPS] +
            [(f'q_l1.7@5-24~{s}', 'v_edit', [f'v_edit.basis={basis_path(s)}', 'v_edit.lam=1.7', 'v_edit.blocks=5-24',
                                             f'v_edit.steps={s}']) for s in STEPS])


def long_modes():
    return [ours(lam, '0-24', LONG) for lam in LONG_LAMS]


def modes():
    return ['original']+[ours(*g) for g in grid()]+[DAVE]+[m for m, _, _ in parts()]+long_modes()+[DAVE10]


def ref_dave(F):
    """The DAVE that rules 1-2 compare with: steps 0-10 where it was run (user 11-05 20:00), else steps 0-19."""
    return DAVE10 if DAVE10 in F else DAVE


def basis_path(steps):
    return f'bases/klein_dev8_s{steps}_k256.pt'


def gen_cmd(split, name, preset, overrides, shard=None):
    cmd = [PY, 'scripts/generate.py', '--model', 'klein', '--dataset', split, '--preset', preset, '--name', name,
           '--gpu', '{gpu}']+overrides+(['--shard', shard] if shard else [])
    return f'cd {ROOT} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '+' '.join(shlex.quote(c) if c != '{gpu}' else c
                                                                                      for c in cmd)


def shards(split):
    return [f'{i}/{COCO_SHARDS}' for i in range(COCO_SHARDS)] if split == 'coco48' else [None]


def jobs():
    basis = [f'{NEED_BASIS}\tcd {ROOT} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} scripts/fit_basis.py '
             f'--model klein --name klein_dev8 --dataset dev8 --steps {" ".join(STEPS)} --gpu {{gpu}}']
    files = dict(basis=basis, basis_long=[       # steps 0-10 (user 11-05 20:00); single set -> name it like the sets
        f'{NEED_BASIS}\tcd {ROOT} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {PY} scripts/fit_basis.py '
        f'--model klein --name {Path(basis_path(LONG)).stem} --dataset dev8 --steps {LONG} --gpu {{gpu}}'])
    for split in SPLITS:                  # one list per split, run in order dev, ho3, coco48: each split is evaluated early
        gen = [f'{NEED_GEN}\t'+gen_cmd(split, 'original', 'original', [], sh) for sh in shards(split)]
        for lam, b, s in grid():
            ov = [f'v_edit.basis={basis_path(s)}', f'v_edit.lam={lam}', f'v_edit.blocks={b}', f'v_edit.steps={s}',
                  f'pad_v.steps={s}', f'pad_v.factor={PAD}']
            gen += [f'{NEED_GEN}\t'+gen_cmd(split, ours(lam, b, s), 'ours', ov, sh) for sh in shards(split)]
        g = EVAL/split/'generation'
        files[f'gen_{split}'] = gen
        if split == 'dev':                # added 11-05 after dev generation had started: run by run_extra_dev.sh
            files['extra_dev'] = [l for l in gen if '@0-24~' in l]+[
                f'{NEED_GEN}\t'+gen_cmd(split, m, preset, ov) for m, preset, ov in parts()]
        files[f'dave_{split}'] = [f'{NEED_GEN}\t'+gen_cmd(split, DAVE, 'dave', [], sh) for sh in shards(split)]
        if split in LONG_SPLITS:          # steps 0-10 (user 11-05 20:00): run by run_steps10.sh
            files[f'dave10_{split}'] = [f'{NEED_GEN}\t'+gen_cmd(split, DAVE10, 'dave', [f'dave.steps={LONG}'])]
            files[f'long_{split}'] = [
                f'{NEED_GEN}\t'+gen_cmd(split, ours(lam, '0-24', LONG), 'ours',
                                        [f'v_edit.basis={basis_path(LONG)}', f'v_edit.lam={lam}', 'v_edit.blocks=0-24',
                                         f'v_edit.steps={LONG}', f'pad_v.steps={LONG}', f'pad_v.factor={PAD}'])
                for lam in LONG_LAMS]
        files[f'eval_{split}'] = [
            f'{NEED_EVAL}\tcd {MIG} && {PY} value_diversity/study_eval.py {g} --name {NAME} --device cuda:{{gpu}} && '
            f'{PY} -m value_diversity.vendi_eval {g} --name {NAME}',
            f'{NEED_JUDGE}\tcd {MIG} && {PY} value_diversity/study_judge.py {g} --name {NAME} --device cuda:{{gpu}}']
    for name, lines in files.items():
        (E/f'jobs_{name}.txt').write_text('\n'.join(lines)+'\n')
        print(f'jobs_{name}.txt: {len(lines)} jobs')


def link(splits=SPLITS):
    """Symlink every complete (split, mode, prompt) into the value_diversity layout; incomplete ones are left out."""
    from div import config, data
    dcfg = config.load('klein').data
    n_link = n_skip = 0
    for split in splits:
        rows = data.prompts(dcfg, split)
        for mode in modes():
            run = OUT/split/mode
            for r in rows:
                files = [run/'images'/r['id']/f'{s}.png' for s in r['seeds']]
                dst = EVAL/split/'generation'/r['id']/mode
                if not all(f.exists() for f in files):
                    n_skip += run.exists()
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                for s, f in zip(r['seeds'], files):
                    p = dst/f'seed{s}.png'
                    if not p.is_symlink():
                        os.symlink(f, p)
                meta = dict(pid=r['id'], prompt=r['prompt'], mode=mode, seeds=r['seeds'], model='klein',
                            run=str(run), source='mmdit_diversity E03 (symlinks)')
                (dst/'meta.json').write_text(json.dumps(meta, indent=1))
                n_link += 1
    print(f'linked {n_link} (split, mode, prompt) folders; {n_skip} incomplete skipped')


# ------------------------------------------------------------------ report
def frames(split):
    import pandas as pd
    root = EVAL/split
    v = pd.read_csv(root/f'eval_{NAME}_vendi.csv').pivot_table(index='pid', columns='mode', values='vendi_dino')
    d = pd.read_csv(root/f'eval_{NAME}_diversity.csv')
    mu = d.pivot_table(index='pid', columns='mode', values='musiq')
    ct = d.pivot_table(index='pid', columns='mode', values='clip_text')
    j = pd.read_csv(root/f'judge_{NAME}_0of1.csv')
    for c in ('p_match', 'p_defect', 'p_style'):
        j[c] = pd.to_numeric(j[c])
    j = j.drop_duplicates(['pid', 'mode', 'seed'])
    j['brk'] = ((j.p_defect >= .5) | (j.p_style >= .5))*100.
    j['mis'] = (j.p_match < .5)*100.
    out = {}
    for mode in v.columns:
        jm = j[j['mode'] == mode].groupby('pid')[['brk', 'mis']].mean()
        if jm.empty:
            continue
        out[mode] = pd.DataFrame(dict(vendi=v[mode], dV=v[mode]-v['original'], musiq=mu[mode], clip_text=ct[mode])).join(jm, how='inner')
    return out


COLS = ('dV', 'brk', 'mis', 'musiq', 'clip_text')


def bs(D, rng, n=2000):
    """D: [prompts, COLS] per-prompt differences -> {col: [mean, 2.5%, 97.5%]} over a prompt bootstrap."""
    s = D[rng.integers(0, len(D), (n, len(D)))].mean(1)
    lo, hi = np.percentile(s, [2.5, 97.5], axis=0)
    return {c: [float(D[:, i].mean()), float(lo[i]), float(hi[i])] for i, c in enumerate(COLS)}


def report():
    rng = np.random.default_rng(0)
    out, lines = {}, ['# E03 klein — 표 (자동 생성, e03.py report)', '',
                      'ΔVendi = 조건 − Original (DINO, prompt 평균). 깨짐 = defect 또는 style ≥ .5, 불일치 = match < .5 (%).',
                      '차이 열은 Original 대비 (DAVE 열은 DAVE 대비), [95% prompt bootstrap].', '']
    for split in SPLITS:
        try:
            F = frames(split)
        except FileNotFoundError as e:
            lines += [f'## {split}: 평가 파일 없음 ({e.filename})', '']
            continue
        if 'original' not in F:
            continue
        pids = sorted(set.intersection(*[set(f.index) for f in F.values()]))
        base = F['original']
        A = {m: F[m].loc[pids, list(COLS)].to_numpy(float) for m in F}
        B = A['original'].copy()
        B[:, 0] = 0.                                   # dV is already relative to Original
        res = {}
        ref = ref_dave(F)
        for mode in [m for m in modes() if m in F and m != 'original']:
            r = bs(A[mode]-B, rng)
            if ref in F and mode not in (DAVE, DAVE10):
                dv = bs(A[mode]-A[ref], rng)
                r['vs_dave'] = dv
                r['R1'] = bool(dv['dV'][0] >= 0)
                r['R2'] = bool(dv['brk'][0] <= 2 and dv['mis'][0] <= 2 and dv['musiq'][0] >= -1)
            res[mode] = r
        out[split] = dict(n_prompts=len(pids), ref_dave=ref, original_vendi=float(base.loc[pids, 'vendi'].mean()),
                          original_brk=float(base.loc[pids, 'brk'].mean()), original_mis=float(base.loc[pids, 'mis'].mean()),
                          original_musiq=float(base.loc[pids, 'musiq'].mean()), modes=res)
        f = lambda x, d=2: f'{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]'
        lines += [f'## {split} ({len(pids)} prompt × 16 seed)', '',
                  f"Original: Vendi {out[split]['original_vendi']:.2f}, 깨짐 {out[split]['original_brk']:.1f}%, "
                  f"불일치 {out[split]['original_mis']:.1f}%, MUSIQ {out[split]['original_musiq']:.1f}", '',
                  f"규칙 1·2의 DAVE 기준: {ref} ({'스텝 0–10, 사용자 11-05 20:00' if ref == DAVE10 else '스텝 0–19'})", '',
                  '| 조건 | ΔVendi | 깨짐 차 | 불일치 차 | MUSIQ 차 | CLIP-text 차 | DAVE 대비 ΔVendi | 규칙 1 | 규칙 2 |',
                  '|---|---|---|---|---|---|---|---|---|']
        for mode, r in res.items():
            vd = r.get('vs_dave')
            lines.append(f"| {mode} | {f(r['dV'])} | {f(r['brk'], 1)} | {f(r['mis'], 1)} | {f(r['musiq'], 2)} | "
                         f"{f(r['clip_text'], 4)} | {f(vd['dV']) if vd else '—'} | "
                         f"{('✔' if r['R1'] else '✘') if vd else '—'} | {('✔' if r['R2'] else '✘') if vd else '—'} |")
        lines.append('')
    held = [s for s in ('ho3', 'coco48') if s in out]   # COCO48 not run (user 11-05 17:50: ho3 only for now)
    passing = [m for m in modes() if held and all(out[s]['modes'].get(m, {}).get('R1') and
                                                  out[s]['modes'][m].get('R2') for s in held)]
    out['pass_splits'], out['pass'] = held, passing
    lines += [f"## 통과 ({'와 '.join(held) or '평가된 split 없음'} 규칙 1·2"
              f"{'' if 'coco48' in held else '; COCO48 미실행 (사용자 11-05)'})", '',
              ', '.join(passing) if passing else '없음 (또는 DAVE 미실행)', '']
    (E/'REPORT_tables.md').write_text('\n'.join(lines))
    (E/'verdict.json').write_text(json.dumps(out, indent=1))
    print('\n'.join(lines))


def missing(split, jobs_file, out_file):
    """Write the job lines of jobs_file whose run (--name) still lacks images of split to out_file."""
    import re
    from div import config, data
    rows = data.prompts(config.load('klein').data, split)
    keep = []
    for line in Path(jobs_file).read_text().splitlines():
        if not line.strip():
            continue
        name = shlex.split(re.search(r"--name ('[^']*'|\S+)", line).group(1))[0]
        run = OUT/split/name
        if not all((run/'images'/r['id']/f'{s}.png').exists() for r in rows for s in r['seeds']):
            keep.append(line)
    Path(out_file).write_text(''.join(l+'\n' for l in keep))
    print(f'{split}: {len(keep)} of the jobs in {jobs_file} still miss images')


def sheets():
    """One prompt per figure (rows = Original, DAVE, the four block/step ranges at lam 1.3 / 1.5 / 1.7), 8 seeds."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=14)
    rows = ['original', DAVE, DAVE10]+long_modes()+[m for m, _, _ in parts()]+[ours(lam, b, s) for s in STEPS
                                                                                  for b in BLOCKS for lam in (1.3, 1.5, 1.7)]
    t, lab = 128, 260
    for split in SPLITS:
        gen = EVAL/split/'generation'
        if not gen.exists():
            continue
        dst = E/'sheets'/split
        dst.mkdir(parents=True, exist_ok=True)
        for pd_ in sorted(gen.iterdir()):
            have = [m for m in rows if (pd_/m/'meta.json').exists()]
            if 'original' not in have:
                continue
            meta = json.loads((pd_/'original'/'meta.json').read_text())
            seeds = meta['seeds'][:8]
            im = Image.new('RGB', (lab+8*t, 24+len(have)*t), 'white')
            d = ImageDraw.Draw(im)
            d.text((4, 4), f"{pd_.name}: {meta['prompt']}"[:150], fill='black', font=font)
            for i, m in enumerate(have):
                d.text((4, 24+i*t+t//2-8), m, fill='black', font=font)
                for j, s in enumerate(seeds):
                    im.paste(Image.open(pd_/m/f'seed{s}.png').convert('RGB').resize((t, t), Image.BICUBIC), (lab+j*t, 24+i*t))
            im.save(dst/f'{pd_.name}.jpg', quality=88)
        print(f'sheets/{split}: {len(list(dst.glob("*.jpg")))} figures')


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('stage', choices=['jobs', 'link', 'report', 'sheets', 'missing'])
    a.add_argument('--split', choices=SPLITS, help='link: this split only; missing: the split to check')
    a.add_argument('--jobs', help='missing: job list to check')
    a.add_argument('--out', help='missing: where to write the unfinished job lines')
    x = a.parse_args()
    if x.stage == 'missing':
        missing(x.split, x.jobs, x.out)
    elif x.stage == 'link':
        link([x.split] if x.split else SPLITS)
    else:
        dict(jobs=jobs, report=report, sheets=sheets)[x.stage]()
