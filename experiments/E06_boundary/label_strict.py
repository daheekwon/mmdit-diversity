"""PROTOCOL_ko.md §21: strict-failure sheet labels (condition-masked).

  python experiments/E06_boundary/label_strict.py sheets sd35       # sheets + key (key.csv: open only after labelling)
  python experiments/E06_boundary/label_strict.py score sd35        # join labels_*.csv with the key -> REPORT_strict_<m>.md
  python experiments/E06_boundary/label_strict.py assign sd35 pass2 rare_zoetrope tune_15 tune_20   # §21.2 assignment
  python experiments/E06_boundary/label_strict.py score sd35 pass2  # labels in strict/pass2/ (§21.2 criterion)
  ... --batch strict_split                                          # another batch: own folder, key and condition list
                                                                    # (BATCH_MODES; default batch 'strict' = MODES)

Sample (§10): the e06 prompts shuffled with random.Random(0), first 25, x 8 seeds, per condition (Original included).
Per prompt, every condition's images are shuffled into sheets of 36 numbered tiles; the sheet header carries the prompt.
Labels: labels_<labeller>.csv with columns id,label,subtype (label 1 = strict failure; subtype S/C/G/N/M).
Output folder: outputs/e06_boundary/<model>/e06/strict/.
"""
import csv
import json
import random
import sys
import textwrap
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
from sheets import font                                                       # noqa: E402
from analyze import load                                                      # noqa: E402

MODES = dict(sd35=['original', 'v_l1.14', 'v_l1.2', 'v_l1.2+fixit', 'v_l1.2+fixtxt', 'v_l1.14+padv0.25', 'dave_a0.5',
                   'padv0', 'padv0.125'],
             flux=['original', 'v_l1.72', 'v_l2.01', 'padv0.25', 'padv0.25+fixit', 'padv0.25+fixtxt', 'v_l1.43+padv0.25', 'v_l1.72+padv0.25',
                   'dave_a0.2', 'padv0', 'padv0.125']+sum(C.e4_names('flux'), []))
BATCH_MODES = dict(strict_split=dict(sd35=['original', 'v_l1.14', 'v_l1.17', 'v_l1.2', 'v_l1.2+readkv', 'v_l1.2+readv',
                                            'v_l1.2+readk', 'v_l1.2+readwkv']))     # §26
BATCH = 'strict'                 # set by --batch
FLAGGED = dict(sd35=['v_l1.14+padv0.25', 'padv0', 'padv0.125'], flux=['padv0', 'padv0.125'])   # MUSIQ flag, else in
GROUPS = 4                       # labellers
OVERLAP = 2                      # prompts labelled by two labellers (agreement)
T, COLS, PER = 200, 6, 36
B = 2000


def out_dir(model):
    return C.gen_root(model, 'e06').parent/BATCH


def modes_of(model):
    return BATCH_MODES[BATCH][model] if BATCH != 'strict' else MODES[model]


def sample_pids():
    ids = [r['id'] for r in C.prompts('e06')]
    random.Random(0).shuffle(ids)
    return ids[:25]


def assignment(pids, overlap=None):
    """Labeller -> prompts: round robin, plus overlap prompts that a second labeller also sees (default: the first
    prompt of the first OVERLAP labellers; §21.2: named prompts, each also given to the next labeller)."""
    a = {g: pids[g::GROUPS] for g in range(GROUPS)}
    for p in overlap or [a[k % GROUPS][0] for k in range(OVERLAP)]:
        g = next(g for g in a if p in a[g])
        a[(g+1) % GROUPS] = a[(g+1) % GROUPS]+[p]
    return a


def assign(model, sub, *overlap):
    out = out_dir(model)
    names = sorted(x.name for x in (out/'sheets').glob('*.jpg'))
    a = assignment(sample_pids(), list(overlap))
    asg = {f'L{g}': [n for p in ps for n in names if n.rsplit('_', 1)[0] == p] for g, ps in a.items()}
    (out/sub).mkdir(exist_ok=True)
    (out/sub/'assignment.json').write_text(json.dumps(dict(overlap=list(overlap), sheets=asg), indent=1))
    print({g: len(v) for g, v in asg.items()})


def sheets(model):
    gen = C.gen_root(model, 'e06')
    out = out_dir(model)
    (out/'sheets').mkdir(parents=True, exist_ok=True)
    prompt = {r['id']: r['prompt'] for r in C.prompts('e06')}
    pids = sample_pids()
    rng = random.Random(f'strict-{model}')
    key, ids = [], list(range(len(pids)*len(modes_of(model))*8))
    rng.shuffle(ids)
    files = {}
    f, fb = font(15), font(17)
    for pid in pids:
        tiles = [(m, s) for m in modes_of(model) for s in C.SEEDS['e06']]
        rng.shuffle(tiles)
        files[pid] = []
        for k in range(0, len(tiles), PER):
            part = tiles[k:k+PER]
            rows = (len(part)+COLS-1)//COLS
            head = textwrap.wrap(f'{pid}: {prompt[pid]}', 110)
            H = 10+22*len(head)
            sheet = Image.new('RGB', (COLS*(T+6)+6, H+rows*(T+24)+6), 'white')
            d = ImageDraw.Draw(sheet)
            for i, line in enumerate(head):
                d.text((6, 4+22*i), line, fill='black', font=fb)
            for j, (m, s) in enumerate(part):
                n = ids.pop()
                key.append(dict(id=n, pid=pid, mode=m, seed=s))
                x, y = 6+(j % COLS)*(T+6), H+(j//COLS)*(T+24)
                p = gen/pid/m/f'seed{s}.png'
                sheet.paste(Image.open(p).convert('RGB').resize((T, T)), (x, y))
                d.text((x+2, y+T+2), f'#{n}', fill='black', font=f)
            name = f'{pid}_{k//PER}.jpg'
            sheet.save(out/'sheets'/name, quality=90)
            files[pid].append(name)
    with (out/'key.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['id', 'pid', 'mode', 'seed'])
        w.writeheader()
        w.writerows(sorted(key, key=lambda r: r['id']))
    asg = {f'L{g}': [s for p in ps for s in files[p]] for g, ps in assignment(pids).items()}
    (out/'assignment.json').write_text(json.dumps(dict(pids=pids, sheets=asg, tiles=len(key)), indent=1))
    print(out, len(key), 'tiles', {g: len(v) for g, v in asg.items()})


def boot(d, rng):
    i = rng.integers(0, len(d), (B, len(d)))
    return float(d.mean()), float(np.percentile(d[i].mean(1), 2.5)), float(np.percentile(d[i].mean(1), 97.5))


def score(model, sub=''):
    out = out_dir(model)
    lab_dir = out/sub if sub else out
    key = {int(r['id']): r for r in csv.DictReader((out/'key.csv').open())}
    lab = {}                                                # id -> {labeller: (label, subtype)}
    for f in sorted(lab_dir.glob('labels_*.csv')):
        for r in csv.DictReader(f.open()):
            lab.setdefault(int(r['id']), {})[f.stem[7:]] = (int(r['label']), r.get('subtype', '').strip())
    missing = [i for i in key if i not in lab]
    if missing:
        raise SystemExit(f'{len(missing)} tiles without a label, e.g. {missing[:10]}')
    # agreement on doubly labelled tiles
    two = [v for v in lab.values() if len(v) >= 2]
    agree = None
    if two:
        a = np.array([[x[0] for x in list(v.values())[:2]] for v in two])
        po = float((a[:, 0] == a[:, 1]).mean())
        pe = float(a[:, 0].mean()*a[:, 1].mean()+(1-a[:, 0].mean())*(1-a[:, 1].mean()))
        agree = (len(two), po, (po-pe)/(1-pe) if pe < 1 else float('nan'))
    # one label per tile: failure if any labeller said failure (conservative against undercounting); report both
    y = {i: max(x[0] for x in v.values()) for i, v in lab.items()}
    styp = {i: ','.join(sorted({x[1] for x in v.values() if x[0] and x[1]})) for i, v in lab.items()}
    pids = sorted({r['pid'] for r in key.values()})
    rate = {(p, m): [] for p in pids for m in modes_of(model)}
    subs = {m: {} for m in modes_of(model)}
    for i, r in key.items():
        rate[(r['pid'], r['mode'])].append(100.*y[i])
        for s in filter(None, styp[i].split(',')):
            subs[r['mode']][s] = subs[r['mode']].get(s, 0)+1
    R = {k: float(np.mean(v)) for k, v in rate.items()}
    rng = np.random.default_rng(0)
    crit = ('실패 = U 알아볼 수 없음(S 실루엣 / C 클로즈업 / G 평면 그래픽 / M 빠짐·잘림) / D 깨짐 / N 붕괴, 화풍은 판정에 쓰지 않음(§21.2)'
            if sub else '실패 = S 실루엣 / C 클로즈업 / G 흑백 평면 그래픽 / N 붕괴 / M 주 물체 빠짐·잘림 (1차, §21)')
    lines = [f'# E06 {model} 엄격 실패 시트 라벨 {sub or "1차"} (PROTOCOL §21, 원본 대비, prompt 짝 bootstrap 95%)', '',
             f'prompt {len(pids)}개 × seed 8 = 조건마다 {8*len(pids)}장. {crit}. 두 라벨러가 본 칸은 한 명이라도 실패로 보면 실패.', '',
             '판독 열은 같은 25 prompt에서 다시 낸 판정 모델 값(원본 대비 pp)이다. 다시 읽은 창(사후) = 엄격 실패 − 원본 ≤ +2pp이고 '
             '불일치 − 원본(50 prompt, 사전 정의) ≤ +2pp.', '',
             '| 조건 | 실패 장수 | 실패율 | 엄격 실패 − 원본 [95%] | 하위 유형 | 판독 결함 또는 화풍 | 판독 결함만 | 판독 불일치 | '
             'MUSIQ 플래그 판정 | 다시 읽은 창 |', '|---|---|---|---|---|---|---|---|---|---|']
    T = load(model)
    allp = sorted(p for (p, mm) in T if mm == 'original')

    def jd(m, k, ps):
        return float(np.mean([T[(p, m)][k]-T[(p, 'original')][k] for p in ps]))
    for m in modes_of(model):
        d = np.array([R[(p, m)]-R[(p, 'original')] for p in pids])
        n = int(sum(y[i] for i, r in key.items() if r['mode'] == m))
        t = boot(d, rng)
        dec = ('창 안' if t[0] <= 2 else '창 밖') if BATCH == 'strict' and m in FLAGGED.get(model, []) else ''
        st = ', '.join(f'{k} {v}' for k, v in sorted(subs[m].items()))
        if m == 'original':
            lines.append(f'| {m} | {n} | {np.mean([R[(p, m)] for p in pids]):.1f}% | — | {st} | — | — | — | | |')
            continue
        mis50 = jd(m, 'mismatch', allp)
        win = '안' if t[0] <= 2 and mis50 <= 2 else '밖'
        lines.append(f'| {m} | {n} | {np.mean([R[(p, m)] for p in pids]):.1f}% | {t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}] | {st} | '
                     f'{jd(m, "broken", pids):+.1f} | {jd(m, "defect", pids):+.1f} | {jd(m, "mismatch", pids):+.1f} '
                     f'(50 prompt {mis50:+.1f}) | {dec} | {win} |')
    if agree:
        lines += ['', f'- 두 라벨러 일치: {agree[0]}칸, 일치율 {agree[1]:.3f}, κ {agree[2]:.2f}']
    rep = lab_dir/f'REPORT_strict_{model}{"_"+sub if sub else ""}.md'
    rep.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--batch' in argv:
        k = argv.index('--batch')
        BATCH = argv[k+1]
        del argv[k:k+2]
    {'sheets': sheets, 'score': score, 'assign': assign}[argv[0]](*argv[1:])
