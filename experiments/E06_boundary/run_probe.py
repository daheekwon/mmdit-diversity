"""E06 stage 2 (PROTOCOL_ko.md §9).

  randbasis --model M            random directions orthogonal to the V-edit basis Q (per block, same k), QR, saved once
  write     --model M --gpu G    open-loop write measure on the dev8 calibration prompts (seeds 40000-40003), no edit
                                 applied: lam_r = 1 + (lam_final - 1) sqrt(sum E_Q / sum E_R)  -> <model>/write_measure.json
  null      --model M --gpu G    null checks (one prompt, two seeds) -> <model>/null_checks.json
  signals   --model M [--modes] [--shard i/n] --gpu G   signal capture, seeds 71000-71001 -> <model>/e06/probe_sig/
  e3        --model M [--modes] [--shard i/n] --gpu G   E3 conditions, seeds 71000-71007 -> <model>/e06/generation/
Existing outputs are skipped.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from div import config, models, splits                                        # noqa: E402
from div.config import resolve                                                # noqa: E402
from div.method import Controller                                             # noqa: E402
import common as C                                                            # noqa: E402
from probe import Probe, concept_index                                        # noqa: E402

RES = [f'model.resolution={C.RES}']


def tokens(pipe, model, prompt, pos):
    if model == 'sd35':
        ci = pipe.tokenizer(prompt, padding='max_length', max_length=models.CLIP_LEN, truncation=True)['input_ids']
        ti = pipe.tokenizer_3(prompt, padding='max_length', max_length=256, truncation=True)['input_ids']
        ids = ci+ti
        toks = [pipe.tokenizer.convert_ids_to_tokens(ids[p]) if p < models.CLIP_LEN else
                pipe.tokenizer_3.convert_ids_to_tokens(ids[p]) for p in pos.tolist()]
    else:
        ids = pipe.tokenizer_2(prompt, padding='max_length', max_length=256, truncation=True)['input_ids']
        toks = [pipe.tokenizer_2.convert_ids_to_tokens(ids[p]) for p in pos.tolist()]
    return toks


def randbasis(model):
    cfg = config.load(model)
    path = resolve(cfg.method.v_edit.basis)
    src = torch.load(path, weights_only=True)
    out, R, check = dict(src), {}, {}
    for b, q in src['Q'].items():
        qd = q.double()
        g = torch.Generator().manual_seed(1000+b)
        G = torch.randn(q.shape, generator=g, dtype=torch.float64)
        for _ in range(2):
            G = G-qd@(qd.T@G)
        r, _ = torch.linalg.qr(G)
        R[b] = r.float()
        check[b] = dict(max_QtR=float((qd.T@r).abs().max()), orth_err=float((r.T@r-torch.eye(r.shape[1], dtype=r.dtype)).abs().max()),
                        q_orth_err=float((qd.T@qd-torch.eye(qd.shape[1], dtype=qd.dtype)).abs().max()))
    out.update(Q=R, unit={b: float('nan') for b in R}, check=check,
               source=f'E06 random directions: per block k Gaussian columns orthogonal to Q of {path}, QR, seed 1000+block')
    dst = C.random_basis(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    worst = {k: max(c[k] for c in check.values()) for k in ('max_QtR', 'orth_err', 'q_orth_err')}
    print(dst, worst, flush=True)


def write(model, gpu):
    rows, seeds = splits.split('dev8')
    seeds = seeds[:4]
    cfg = config.load(model, 'original', RES)
    lam_f = float(config.load(model, 'ours').method.v_edit.lam)
    pipe = models.load(cfg.model, gpu)
    tot = {}
    for r in rows:
        emb = models.encode(pipe, cfg.model, r['prompt'])
        ctrl = Probe(pipe, cfg, r['prompt'], seeds, write_basis=str(C.random_basis(model)))
        models.generate(pipe, cfg.model, emb, seeds, ctrl, last_step=max(ctrl.vsteps))
        for b, e in ctrl.energy.items():
            t = tot.setdefault(b, [0., 0., 0.])
            for j in range(3):
                t[j] += e[j]
        print('WRITE', r['id'], flush=True)
    eq, er = sum(t[0] for t in tot.values()), sum(t[1] for t in tot.values())
    ratio = math.sqrt(eq/er)
    res = dict(lam_final=lam_f, ratio_Q_over_R=ratio, lam_random=1+(lam_f-1)*ratio,
               per_block={b: dict(E_Q=t[0], E_R=t[1], E_x=t[2], ratio=math.sqrt(t[0]/t[1])) for b, t in sorted(tot.items())},
               prompts=[r['id'] for r in rows], seeds=seeds, steps=sorted(ctrl.vsteps), blocks=ctrl.vblocks,
               note='open loop on the unedited trajectory; E = sum over CFG rows, tokens, channels of (gate * W_out A dV)^2 at unit (lam - 1)')
    C.write_json(model).parent.mkdir(parents=True, exist_ok=True)
    C.write_json(model).write_text(json.dumps(res, indent=1))
    print('LAM_RANDOM', res['lam_random'], 'ratio', ratio, flush=True)


def _diff(a, b):
    a, b = np.asarray(a, dtype=np.int16), np.asarray(b, dtype=np.int16)
    d = np.abs(a-b)
    return dict(max=int(d.max()), mean=float(d.mean()), frac_changed=float((d > 0).mean()))


def null(model, gpu, pid):
    row = next(r for r in C.prompts('e06') if r['id'] == pid)
    seeds = C.SIGNAL_SEEDS
    pipe = models.load(config.load(model, 'original', RES).model, gpu)
    cfg0 = config.load(model, 'original', RES)
    cfg1 = config.load(model, 'v_edit', RES+['v_edit.lam=1.0'])
    cfgp = config.load(model, 'pad_v', RES+['pad_v.factor=1.0'])
    emb = models.encode(pipe, cfg0.model, row['prompt'])
    run = lambda cfg, ctrl: models.generate(pipe, cfg.model, emb, seeds, ctrl)
    ref = run(cfg0, None)
    out = dict(prompt=row['prompt'], seeds=seeds)
    pos = concept_index(pipe, model, row['prompt'])
    out['concept_tokens'] = tokens(pipe, model, row['prompt'], pos)
    cases = dict(
        controller_lam1=lambda: Controller(pipe, cfg1, row['prompt'], seeds),
        signals_original=lambda: Probe(pipe, cfg0, row['prompt'], seeds, signal_steps=C.SIGNAL_STEPS[model]),
        e3_lam1_fixtxt=lambda: Probe(pipe, cfg1, row['prompt'], seeds, twin=('txt',)),
        e3_lam1_fixit=lambda: Probe(pipe, cfg1, row['prompt'], seeds, twin=('it',)),
        e3_lam1_fixboth=lambda: Probe(pipe, cfg1, row['prompt'], seeds, twin=('it', 'txt')),
        e3_pad1_fixboth=lambda: Probe(pipe, cfgp, row['prompt'], seeds, twin=('it', 'txt')))
    imgs = {}
    for name, make in cases.items():
        cfg = cfg0 if name == 'signals_original' else (cfgp if 'pad1' in name else cfg1)
        t0 = time.monotonic()
        ctrl = make()
        imgs[name] = run(cfg, ctrl)
        out[name] = dict(vs_original=[_diff(a, b) for a, b in zip(imgs[name], ref)], seconds=time.monotonic()-t0)
        if name.startswith('e3'):
            out[name]['vs_controller_lam1'] = [_diff(a, b) for a, b in zip(imgs[name], imgs['controller_lam1'])]
        if name == 'signals_original':
            sg = ctrl.signals()[0]
            out[name]['signal_keys'] = {k: len(v) for k, v in sg.items()}
            it = next(iter(sg['it'].values()))
            out[name]['it_shape'] = list(it.shape)
            out[name]['concept_mass_step0_block0'] = float(sg['it'][f"{C.SIGNAL_STEPS[model][0]},0"].float().sum(-1).mean())
        print('NULL', name, json.dumps(out[name])[:400], flush=True)
    dst = C.OUT/model/'null_checks.json'
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    bad = [k for k in cases if any(d['max'] for d in out[k]['vs_original'])]
    print('NULL_DONE', dst, 'FAILED' if bad else 'PASS', bad, flush=True)
    if bad:
        raise SystemExit(1)


def nulldave(model, gpu, pid):
    """§17 null checks for DAVE with the E3 blocks. Must be pixel-identical: DAVE alpha 1 + both blocks vs Original, and
    the probe with DAVE (no block) vs the package Controller with DAVE. Must differ (the twin pass is unedited, so the
    block acts): DAVE standard + both blocks vs DAVE standard. Controller.verify() also fails if the twin pass ran DAVE."""
    row = next(r for r in C.prompts('e06') if r['id'] == pid)
    seeds = C.SIGNAL_SEEDS
    cfg0 = config.load(model, 'original', RES)
    cfgd1 = config.load(model, 'dave', RES+['dave.alpha=1.0'])
    cfgd = config.load(model, 'dave', RES)
    pipe = models.load(cfg0.model, gpu)
    emb = models.encode(pipe, cfg0.model, row['prompt'])
    run = lambda cfg, ctrl: models.generate(pipe, cfg.model, emb, seeds, ctrl)
    ref, dref = run(cfg0, None), run(cfgd, Controller(pipe, cfgd, row['prompt'], seeds))
    out = dict(prompt=row['prompt'], seeds=seeds, dave=dict(cfgd.method.dave))
    cases = dict(dave1_fixboth=(cfgd1, ('it', 'txt'), ref, 'same'), dave_probe_nofix=(cfgd, (), dref, 'same'),
                 dave_fixboth=(cfgd, ('it', 'txt'), dref, 'differ'), dave_fixtxt=(cfgd, ('txt',), dref, 'differ'),
                 dave_fixit=(cfgd, ('it',), dref, 'differ'))
    bad = []
    for name, (cfg, twin, against, want) in cases.items():
        t0 = time.monotonic()
        ctrl = Probe(pipe, cfg, row['prompt'], seeds, twin=twin)
        imgs = run(cfg, ctrl)
        d = [_diff(a, b) for a, b in zip(imgs, against)]
        same = not any(x['max'] for x in d)
        out[name] = dict(want=want, diff=d, twin_steps=sorted(ctrl.twin_steps), dave_calls=len(ctrl.base),
                         seconds=time.monotonic()-t0)
        if same != (want == 'same'):
            bad.append(name)
        print('NULL', name, json.dumps(out[name])[:400], flush=True)
    dst = C.OUT/model/'null_checks_dave.json'
    dst.write_text(json.dumps(out, indent=1))
    print('NULL_DONE', dst, 'FAILED' if bad else 'PASS', bad, flush=True)
    if bad:
        raise SystemExit(1)


def nullsplit(model, gpu, pid):
    """DESIGN_e3_split_ko.md A / B0 null checks. Must be pixel-identical: lam 1 + each read-level fix vs Original; the
    probe with no fix at the split lam vs the package Controller. Must differ: split lam + each fix vs split lam.
    Reported only: split lam + readkv vs the text-stream fix (same manipulation, float32 correction vs bf16 rerun).
    B0 capture: lam 1 + text-stream fix with text signals must equal Original and hold the new signal keys."""
    row = next(r for r in C.prompts('e06') if r['id'] == pid)
    seeds = C.SIGNAL_SEEDS
    cfg0 = config.load(model, 'original', RES)
    cfg1 = config.load(model, 'v_edit', RES+['v_edit.lam=1.0'])
    p, o, _ = C.all_modes(model)[C.SPLIT_LAM[model]]
    cfgl = config.load(model, p, RES+list(o))
    pipe = models.load(cfg0.model, gpu)
    emb = models.encode(pipe, cfg0.model, row['prompt'])
    run = lambda cfg, ctrl: models.generate(pipe, cfg.model, emb, seeds, ctrl)
    ref = run(cfg0, None)
    lref = run(cfgl, Controller(pipe, cfgl, row['prompt'], seeds))
    out = dict(prompt=row['prompt'], seeds=seeds, lam=C.SPLIT_LAM[model])
    cases = {f'lam1_{s}': (cfg1, tw, ref, 'same', {}) for s, tw in C.SPLIT.items()}
    cases['lam_probe_nofix'] = (cfgl, (), lref, 'same', {})
    cases.update({f'lam_{s}': (cfgl, tw, lref, 'differ', {}) for s, tw in C.SPLIT.items()})
    cases['lam1_fixtxt_textsig'] = (cfg1, ('txt',), ref, 'same', dict(signal_steps=C.SIGNAL_STEPS[model], text_sig=True))
    bad, imgs = [], {}
    for name, (cfg, twin, against, want, kw) in cases.items():
        t0 = time.monotonic()
        ctrl = Probe(pipe, cfg, row['prompt'], seeds, twin=twin, **kw)
        imgs[name] = run(cfg, ctrl)
        d = [_diff(a, b) for a, b in zip(imgs[name], against)]
        same = not any(x['max'] for x in d)
        out[name] = dict(want=want, diff=d, twin=list(twin), twin_steps=sorted(ctrl.twin_steps),
                         seconds=time.monotonic()-t0)
        if kw.get('text_sig'):
            sg = ctrl.signals()[0]
            k0 = f"{C.SIGNAL_STEPS[model][0]},0"
            out[name]['signal_shapes'] = {kind: list(sg[kind][k0].shape) for kind in ('kc', 'vc', 'ko', 'vo')}
            out[name]['signal_keys'] = {kind: len(v) for kind, v in sg.items()}
            ok = all(len(sg[kind]) == len(sg['kc']) > 0 for kind in ('vc', 'ko', 'vo'))
            if not ok:
                bad.append(name+':signals')
        if same != (want == 'same'):
            bad.append(name)
        print('NULL', name, json.dumps(out[name])[:400], flush=True)
    fx = Probe(pipe, cfgl, row['prompt'], seeds, twin=('txt',))
    out['lam_readkv_vs_fixtxt'] = [_diff(a, b) for a, b in zip(imgs['lam_readkv'], run(cfgl, fx))]
    print('NULL lam_readkv_vs_fixtxt', json.dumps(out['lam_readkv_vs_fixtxt']), flush=True)
    dst = C.OUT/model/'null_checks_split.json'
    dst.write_text(json.dumps(out, indent=1))
    print('NULL_DONE', dst, 'FAILED' if bad else 'PASS', bad, flush=True)
    if bad:
        raise SystemExit(1)


def generate(model, gpu, kind, names, shard, batch, tag='', text_sig=False):
    table = C.all_modes(model)
    names = names or (C.signal_modes(model) if kind == 'signals' else [m for m in table if '+fix' in m])
    seeds = C.SIGNAL_SEEDS if kind == 'signals' else C.SEEDS['e06']
    root = C.OUT/model/'e06'/(('probe_sig' if kind == 'signals' else 'generation')+(f'_{tag}' if tag else ''))
    if text_sig and kind != 'signals':
        raise SystemExit('--text-sig is for signals')
    rows = C.prompts('e06')
    i, n = map(int, shard.split('/'))
    rows = rows[i::n]
    pipe = models.load(config.load(model, 'original', RES).model, gpu)
    print(f'{root}: {len(rows)} prompts, modes {names}', flush=True)
    for r in rows:
        emb = models.encode(pipe, config.load(model, 'original', RES).model, r['prompt'])
        for mode in names:
            preset, over, twin = table[mode]
            cfg = config.load(model, preset, RES+list(over))
            folder = root/r['id']/mode
            done = lambda s: (folder/f'seed{s}.png').exists() and (kind != 'signals' or (folder/f'sig_seed{s}.pt').exists())
            todo = [s for s in seeds if not done(s)]
            if not todo:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            for k in range(0, len(todo), batch):
                part = todo[k:k+batch]
                if kind == 'signals':          # twin: the E3 / split fixes (signal capture of a blocked run, B0)
                    ctrl = Probe(pipe, cfg, r['prompt'], part, signal_steps=C.SIGNAL_STEPS[model], twin=twin,
                                 text_sig=text_sig)
                else:
                    ctrl = Probe(pipe, cfg, r['prompt'], part, twin=twin)
                images = models.generate(pipe, cfg.model, emb, part, ctrl)
                sigs = ctrl.signals() if kind == 'signals' else [None]*len(part)
                for s, im, sg in zip(part, images, sigs):
                    im.save(folder/f'seed{s}.png')
                    if sg is not None:
                        torch.save(sg, folder/f'sig_seed{s}.pt')
            seen = sorted(int(p.stem[4:]) for p in folder.glob('seed*.png'))
            meta = dict(pid=r['id'], prompt=r['prompt'], group=r['group'], mode=mode, seeds=seen, preset=preset,
                        overrides=list(over), twin=list(twin), resolution=C.RES, batch=batch,
                        concept_pos=ctrl.concept.tolist(), concept_tokens=tokens(pipe, model, r['prompt'], ctrl.concept),
                        signal_steps=list(C.SIGNAL_STEPS[model]) if kind == 'signals' else [], text_sig=text_sig,
                        method={c: dict(cfg.method[c]) for c in config.COMPONENTS if cfg.method[c].enabled},
                        info=ctrl.info(), seconds=time.monotonic()-t0)
            (folder/'meta.json').write_text(json.dumps(meta, indent=1, default=str))
            print(f"SAVED {r['id']} {mode} {len(todo)} images {meta['seconds']:.1f}s", flush=True)
    print('GEN_COMPLETE', flush=True)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('kind', choices=['randbasis', 'write', 'null', 'nulldave', 'nullsplit', 'signals', 'e3'])
    a.add_argument('--model', required=True, choices=['sd35', 'flux'])
    a.add_argument('--gpu', type=int, default=0)
    a.add_argument('--modes', nargs='*')
    a.add_argument('--shard', default='0/1')
    a.add_argument('--pid', default='comp_03')
    a.add_argument('--batch', type=int, default=8)
    a.add_argument('--tag', default='', help='signals / e3: write to probe_sig_<tag> / generation_<tag>')
    a.add_argument('--text-sig', action='store_true', help='signals: also record text V and the other text tokens (B0)')
    x = a.parse_args()
    if x.kind == 'randbasis':
        randbasis(x.model)
    elif x.kind == 'write':
        write(x.model, x.gpu)
    elif x.kind == 'null':
        null(x.model, x.gpu, x.pid)
    elif x.kind == 'nulldave':
        nulldave(x.model, x.gpu, x.pid)
    elif x.kind == 'nullsplit':
        nullsplit(x.model, x.gpu, x.pid)
    else:
        generate(x.model, x.gpu, x.kind, x.modes, x.shard, x.batch, x.tag, x.text_sig)


if __name__ == '__main__':
    main()
