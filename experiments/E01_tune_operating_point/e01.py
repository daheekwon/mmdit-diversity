"""E01 helpers (PROTOCOL_ko.md). Called by run.sh; paths are relative to mmdit_diversity/.

  coco-sets          write the COCO test and tune caption files (data.coco.prompts, data.coco_tune.prompts) once
  check-calib FILE   calibration prompts must not repeat any tune / test prompt or any prompt of the earlier
                     value_diversity runs (every generation meta.json there)
  jobs basis FILE | tune | full   job lines for scripts/lanes.py ("<GiB>\\t<command>"); basis needs the calibration file
  report             section-5 tables (tune_<model>.md) and side-by-side sheets (sheets/)
"""
import json
import random
import sys
from pathlib import Path
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from div import analysis, config, data, splits                                # noqa: E402

E = Path(__file__).resolve().parent
PY = '/NHNHOME/workspace/miniforge3/envs/flowlab/bin/python'
DCFG = OmegaConf.load(ROOT/'configs'/'data.yaml')
COCO = Path('/NHNHOME/workspace/data/coco')
TEST_SRC = COCO/'captions_val2017_img500_cap500.json'      # the 500-caption file of the DAVE paper scale (user, 11-04)
TUNE_SRC = COCO/'captions_val2017_tune50.json'             # 11-04 tune file, before the COCO 48 replacement
MODELS = ('flux', 'sd35')                         # FLUX first: its jobs are the longest
NEED = dict(sd35=22, flux=32)                     # GiB free before a job starts
FIT = dict(sd35=('sd35_calib', '0-11 0-3'), flux=('flux_calib_k256', '0-5'))     # name, step sets (PROTOCOL 8-8)
BASIS = dict(sd35='bases/sd35_calib_s0-11_k256.pt', flux='bases/flux_calib_k256.pt')  # operating point
REF = dict(sd35='bases/sd35_calib_s0-3_k256.pt')  # tune reference only: the steps of the legacy SD3.5 basis
LEGACY = dict(sd35='bases/sd35_legacy_dev8_b0to23_k256.pt', flux='bases/flux_legacy_dev8_k256.pt')
CURVE = dict(sd35=[1.07, 1.105, 1.175, 1.21], flux=[1.36, 1.54, 1.90, 2.08])    # lam; operating point lam-1 x .5/.75/1.25/1.5
TUNE = ('coco_tune', 'imagenet_tune')
FULL = ('imagenet', 'coco')
METRICS = ['vendi_dino', 'vendi_clip', 'lpips', 'clip_score', 'pickscore', 'aesthetic', 'image_reward']


def conditions(model, stage):
    """(run name, preset, overrides)."""
    b = f'v_edit.basis={BASIS[model]}'
    base = [('original', 'original', []), ('dave', 'dave', []), ('cads', 'cads', []), ('ours', 'ours', [b])]
    if stage == 'full':
        return base
    ref = [('ours_fit0-3', 'ours', [f'v_edit.basis={REF[model]}'])] if model in REF else []
    return base+[('pad_v', 'pad_v', []), ('ours_legacy', 'ours', [f'v_edit.basis={LEGACY[model]}'])]+ref + \
        [('ours_head', 'ours_head', [b])] + \
        [(f'ours_l{lam}', 'ours', [b, f'v_edit.lam={lam}']) for lam in CURVE[model]]


def jobs(stage, calib=None):
    if stage == 'basis':
        for m in MODELS:
            name, steps = FIT[m]
            print(f'{NEED[m]}\t{PY} scripts/fit_basis.py --model {m} --name {name} --prompts {Path(calib).resolve()} '
                  f'--steps {steps} --gpu {{gpu}}')
        return
    missing = [p for p in [*BASIS.values(), *REF.values()] if not (ROOT/p).exists()]
    if missing:
        raise SystemExit(f'fit the bases first (run.sh basis): missing {missing}')
    for m in MODELS:
        for d in (TUNE if stage == 'tune' else FULL):
            for name, preset, ov in conditions(m, stage):
                run = f'outputs/{m}/{d}/{name}'
                print(f'{NEED[m]}\t{PY} scripts/generate.py --model {m} --dataset {d} --name {name} --preset {preset} '
                      f'{" ".join(ov)} --gpu {{gpu}} && {PY} scripts/evaluate.py {run} --device cuda:{{gpu}}')


def _write(path, out):
    text = json.dumps(out)
    if path.exists() and path.read_text() != text:
        raise SystemExit(f'{path} exists with different content; not overwriting')
    path.write_text(text)


def coco_sets():
    """The COCO test and tune caption files of PROTOCOL section 3 (data.coco.prompts, data.coco_tune.prompts).
    1. tune: 50 val2017 images outside the 500-image test file TEST_SRC, one caption each (random.Random(0)).
    2. Images of the COCO 48 split (value_diversity, where the operating lam was chosen) are replaced in both, in place
       (same row, so every other prompt keeps its seeds): a val2017 image outside test, tune and COCO 48, one caption
       each, random.Random(1)."""
    full = json.loads((Path(DCFG.coco.real_dir).parents[1]/'annotations'/'captions_val2017.json').read_text())
    src = json.loads(TEST_SRC.read_text())
    caps = {}
    for a in full['annotations']:
        caps.setdefault(a['image_id'], []).append(a)
    caps = {i: sorted(c, key=lambda a: a['id']) for i, c in caps.items()}
    images = {i['id']: i for i in full['images']}
    test = [a['image_id'] for a in src['annotations']]
    pool = sorted(i for i in images if i not in set(test))
    rng = random.Random(0)
    pick = sorted(rng.sample(pool, 50))
    tune = [rng.choice(caps[i]) for i in pick]
    if TUNE_SRC.exists():                                     # the 11-04 tune file is this step's output
        assert [a['id'] for a in tune] == [a['id'] for a in json.loads(TUNE_SRC.read_text())['annotations']]
    c48 = {r['image_id'] for r in splits.split('coco48')[0]}
    rng = random.Random(1)
    free = sorted(i for i in images if i not in c48 | set(test) | set(pick))
    log = []
    for name, anns in (('test', src['annotations']), ('tune', tune)):
        for k, a in enumerate(anns):
            if a['image_id'] in c48:
                new = free.pop(rng.randrange(len(free)))
                anns[k] = rng.choice(caps[new])
                log.append(dict(set=name, row=k, old_image=a['image_id'], old_caption=a['caption'].strip(),
                                new_image=new, new_caption=anns[k]['caption'].strip()))
    note = (f' COCO 48 images replaced in place (mmdit_diversity E01, random.Random(1)): '
            + '; '.join(f"{r['set']} row {r['row']} image {r['old_image']} -> {r['new_image']}" for r in log))
    for path, anns, desc in ((Path(DCFG.coco.prompts), src['annotations'], src['info'].get('description', '')),
                             (Path(DCFG.coco_tune.prompts), tune, 'COCO 2017 val: 50 images outside the 500-image test '
                              'file, one caption each (mmdit_diversity E01 tune set, random.Random(0))')):
        ids = [a['image_id'] for a in anns]
        _write(path, dict(info=dict(full['info'], description=desc+note), licenses=full['licenses'],
                          images=[images[i] for i in ids], annotations=anns))
        print(f'{path}: {len(anns)} captions')
    for r in log:
        print(f"  {r['set']} row {r['row']}: {r['old_image']} '{r['old_caption']}' -> {r['new_image']} '{r['new_caption']}'")
    (E/'coco_replaced.json').write_text(json.dumps(log, indent=1, ensure_ascii=False))


def norm(p):
    return ' '.join(p.lower().strip().rstrip('.').split())


def check_calib(file):
    calib = data.calibration_prompts(OmegaConf.merge(DCFG, {'calibration': {'prompts': file}}))
    used = {norm(r['prompt']): d for d in data.DATASETS for r in data.prompts(DCFG, d)}
    for meta in (ROOT.parent/'value_diversity'/'results').rglob('meta.json'):
        try:
            p = json.loads(meta.read_text()).get('prompt')
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(p, str):
            used.setdefault(norm(p), 'value_diversity')
    hits = [(p, used[norm(p)]) for p in calib if norm(p) in used]
    dup = len(calib)-len({norm(p) for p in calib})
    print(f'{len(calib)} calibration prompts; {len(used)} other prompts checked; overlaps {len(hits)}; repeats {dup}')
    for p, where in hits:
        print(f'  overlap ({where}): {p}')
    if hits or dup:
        raise SystemExit(1)


def v_change(model):
    """Relative image-V change of the operating lam with each basis: (lam - 1) x mean unit over the edited blocks."""
    v = config.load(model, 'ours').method.v_edit
    blocks = config.parse_range(v.blocks, dict(sd35=24, flux=57)[model])
    out = []
    for name, path in (('새 기저', BASIS[model]), ('새 기저 step 0–3 적합 (참고)', REF.get(model)), ('이전 기저', LEGACY[model])):
        if path is None:
            continue
        f = (ROOT/path).with_suffix('.json')
        if f.exists():
            unit = json.loads(f.read_text())['unit']
            u = sum(unit[str(b)] for b in blocks)/len(blocks)
            out.append(f'{name}: λ{v.lam} → image V 상대 변화량 {(v.lam-1)*u:.4f} (block 단위 평균 {u:.4f})')
    return out


def report():
    import subprocess
    (E/'sheets').mkdir(exist_ok=True)
    for m in MODELS:
        lines = [f'# E01 tune: {m} (Original 대비, prompt 단위 짝지은 차이, 95% bootstrap 구간)', '', *v_change(m), '']
        for d in TUNE:
            out = ROOT/'outputs'/m/d
            runs = [out/n for n, _, _ in conditions(m, 'tune') if n != 'original' and (out/n/'metrics').exists()]
            if not (out/'original'/'metrics').exists() or not runs:
                lines += [f'## {d}', '', '(평가 결과 없음)', '']
                continue
            t = analysis.compare(out/'original', runs, METRICS)
            v = t[(t.metric == 'vendi_dino') & (t.run == 'ours')]
            check = 'PASS' if len(v) and v.lo.iloc[0] > 0 else 'FAIL'
            lines += [f'## {d}', '', f'§5-1 새 기저 운영점 Vendi(DINO) 증가의 95% 구간 > 0: **{check}**', '',
                      t.to_markdown(index=False, floatfmt='.4g'), '']
            pairs = [('ours_legacy', 'ours', '새 기저 운영점 − 이전 기저 운영점'),
                     ('ours_legacy', 'ours_fit0-3', '새 기저 step 0–3 적합 − 이전 기저 (보정 prompt만 다름)'),
                     ('ours_fit0-3', 'ours', '새 기저 step 0–11 적합 − step 0–3 적합 (적합 step만 다름)')]
            for a, b, title in pairs:
                if (out/a/'metrics').exists() and (out/b/'metrics').exists():
                    t2 = analysis.compare(out/a, [out/b], METRICS)
                    lines += [title+':', '', t2.to_markdown(index=False, floatfmt='.4g'), '']
            rows = ['original', 'ours_legacy'] + (['ours_fit0-3'] if m in REF else []) + ['ours']
            subprocess.run([PY, str(ROOT/'scripts'/'sheet.py'), 'side', *(str(out/n) for n in rows),
                            '--prompts', '10', '--out', str(E/'sheets'/f'{m}_{d}')], check=True)
        (E/f'tune_{m}.md').write_text('\n'.join(lines))
        print(f'wrote {E/f"tune_{m}.md"}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'coco-sets':
        coco_sets()
    elif cmd == 'check-calib':
        check_calib(sys.argv[2])
    elif cmd == 'jobs':
        jobs(sys.argv[2], *sys.argv[3:4])
    elif cmd == 'report':
        report()
    else:
        raise SystemExit(__doc__)
