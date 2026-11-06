"""DESIGN_e3_split_ko.md B0: how much the text tokens still change against Original when the text route is blocked
within each step (the text stream's reaction to an image changed through the image path).

  python experiments/E06_boundary/split_b0.py sd35

Input: <model>/e06/probe_sig_b0/<pid>/<mode>/sig_seed<s>.pt (run_probe.py signals --tag b0 --text-sig), modes
Original, the split lam (edit only), lam + text-stream fix, lam + concept-column fix; seeds 71000-71001.
Per image pair (same prompt, seed) and signal step: 1 - cosine with Original of each text token's 128-d projection,
mean over tokens, then over blocks. Kinds: prompt-word K (kc), prompt-word V (vc), other text tokens' K (ko) and V (vo).
Prompt bootstrap (2,000, seed 0) of per-prompt means. Rule fixed before data (design B0): remaining ratio =
change(text-stream fix) / change(edit only) at the last edited step, prompt-word K; >= 0.5 -> propose B1, else no B1.
Writes <model>/e06/REPORT_B0_<model>.md.
"""
import sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402

B = 2000
KINDS = [('kc', '단어 토큰 K'), ('vc', '단어 토큰 V'), ('ko', '그 밖의 텍스트 토큰 K'), ('vo', '그 밖의 텍스트 토큰 V')]


def change(e, o, kind, step, nb):
    keys = [f'{step},{b}' for b in range(nb)]
    a = torch.stack([e[kind][k].float() for k in keys])                      # [nb, ntok, 128]
    z = torch.stack([o[kind][k].float() for k in keys])
    return float((1-torch.nn.functional.cosine_similarity(a, z, dim=-1)).mean())


def boot(x, rng):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    model = sys.argv[1]
    rng = np.random.default_rng(0)
    root = C.OUT/model/'e06'/'probe_sig_b0'
    lam = C.SPLIT_LAM[model]
    modes = [lam, f'{lam}+fixtxt', f'{lam}+fixit']
    steps = C.SIGNAL_STEPS[model]
    nb = len([k for k in torch.load(next(root.glob('*/original/sig_seed*.pt')), weights_only=True)['kc'] if k.startswith(f'{steps[0]},')])
    pids = sorted(p.name for p in root.iterdir()
                  if all((p/m/f'sig_seed{s}.pt').exists() for m in ['original']+modes for s in C.SIGNAL_SEEDS))
    R = {}                                                                    # (mode, kind, step) -> per-prompt array
    for pid in pids:
        for s in C.SIGNAL_SEEDS:
            o = torch.load(root/pid/'original'/f'sig_seed{s}.pt', weights_only=True)
            for m in modes:
                e = torch.load(root/pid/m/f'sig_seed{s}.pt', weights_only=True)
                for kind, _ in KINDS:
                    for st in steps:
                        R.setdefault((m, kind, st), {}).setdefault(pid, []).append(change(e, o, kind, st, nb))
    R = {k: np.array([np.mean(v[p]) for p in pids]) for k, v in R.items()}
    last = steps[-2]                                                          # last edited step (SIGNAL_STEPS order)
    out = [f'# E06 {model} B0: 텍스트 토큰의 원본 대비 변화 (DESIGN_e3_split_ko.md, 결과를 본 뒤 추가한 측정)', '',
           f'prompt {len(pids)}개 × seed {len(C.SIGNAL_SEEDS)}, {lam}. 값 = 1 − 코사인(128차원 사영, 토큰 평균 → 블록 평균), '
           f'[prompt bootstrap 95%]. step {steps[-1]}은 편집하지 않는 첫 step이다.', '']
    for kind, name in KINDS:
        out += [f'## {name}', '', '| 조건 | '+' | '.join(f'step {s}' for s in steps)+' |', '|'+'---|'*(len(steps)+1)]
        for m in modes:
            out.append(f'| {m} | '+' | '.join('{:.4f} [{:.4f}, {:.4f}]'.format(*boot(R[(m, kind, s)], rng)) for s in steps)+' |')
        out.append('')
    out += [f'## 남은 비율 = 텍스트 스트림 고정 / 편집만 (같은 재표집)', '']
    rows = {}
    for kind, name in KINDS:
        cells = []
        for s in steps:
            a, b = R[(f'{lam}+fixtxt', kind, s)], R[(lam, kind, s)]
            bs = []
            for _ in range(B):
                i = rng.integers(0, len(pids), len(pids))
                bs.append(a[i].mean()/b[i].mean())
            t = (a.mean()/b.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
            rows[(kind, s)] = t
            cells.append('{:.2f} [{:.2f}, {:.2f}]'.format(*t))
        out.append(f'- {name}: '+', '.join(f'step {s} {c}' for s, c in zip(steps, cells)))
    t = rows[('kc', last)]
    dec = 'B1 제안 (반응이 크다)' if t[0] >= .5 else 'B1 하지 않음 (짝 기준 차단이 텍스트 변화의 대부분을 이미 없앴다)'
    out += ['', f'**판정 (설계 B0, 결과를 보기 전에 고정).** 단어 토큰 K, step {last}: 남은 비율 {t[0]:.2f} '
            f'[{t[1]:.2f}, {t[2]:.2f}] → {dec}.']
    rep = C.OUT/model/'e06'/f'REPORT_B0_{model}.md'
    rep.write_text('\n'.join(out)+'\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
