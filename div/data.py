"""Prompt lists and real-image sets.

imagenet  1,000 class-label prompts: template.format(first synonym of the class: devkit meta.mat, or the HF classes.py
          when devkit is null), classes in sorted-WNID order (index 0 = n01440764). Real set: the first `real_per_class`
          validation files (by name) of each class (10 x 1000 = 10K): from the devkit ground truth over a flat val_dir,
          or from val_dir/<wnid>/ folders.
coco      the prompts in the file `coco.prompts`, in file order (txt: one caption per line; json: a COCO caption file
          with an `annotations` list, or a list of strings or of {id, caption}; jsonl: one {id, caption} per line).
          Prompt id = coco_<caption annotation id>; surrounding whitespace is stripped. Real set: every image of
          val2017 (5K).
imagenet_tune, coco_tune  tune sets with `base: imagenet | coco`: the base dataset's prompt rules and real set, their own
          seed_base (so no tune image is a test image); imagenet_tune keeps the first n_classes classes of the fixed shuffle.
vbench, vbench_tune  video prompts, one per line (`prompts`), with the short originals line by line (`short_prompts`,
          kept in the row as `short`). Prompt id = <id_prefix><line>[_<category>], category = line // per_category.
          vbench = the 800 category prompts; vbench_tune = the 946 dimension prompts (no overlap with the 800), first
          n_subset of the fixed shuffle. No real set.
dev, dev8, ho3, coco48  fixed splits with their own seed lists (div/splits.py; `split: <name>` in data.yaml): sample j
          of every prompt gets seeds[j]. No real set.
Seeds: sample j of prompt i gets seed_base + 100 i + j for every method, so runs are paired image by image.
A tune subset (n_prompts) takes the first n of a fixed shuffle of the prompts and keeps their full-run seeds,
so tune images are the same files as in the full run.
"""
import json
import random
from pathlib import Path

SEED_STRIDE = 100
DATASETS = ('imagenet', 'coco', 'imagenet_tune', 'coco_tune', 'vbench', 'vbench_tune', 'dev', 'dev8', 'ho3', 'coco48')


HF_RENAMED = {'n03126707': 'crane'}     # the HF list renames the second 'crane' (the machine) to 'crane2'; devkit: 'crane'


def imagenet_classes(c):
    """(sorted WNIDs, {wnid: first synonym}, {ILSVRC2012 id: wnid} or None) from c.devkit (meta.mat) or, when devkit is
    null, from c.classes (HF imagenet-1k classes.py: its IMAGENET2012_CLASSES dict literal, read with ast, not run)."""
    if c.get('devkit'):
        from scipy.io import loadmat
        syn = loadmat(Path(c.devkit)/'data'/'meta.mat', squeeze_me=True)['synsets']
        leaf = [s for s in syn if int(s['num_children']) == 0]
        by_id = {int(s['ILSVRC2012_ID']): str(s['WNID']) for s in leaf}
        words = {str(s['WNID']): str(s['words']) for s in leaf}
    else:
        words, by_id = _hf_classes(Path(c.classes)), None
    if len(words) != 1000:
        raise ValueError(f'expected 1000 leaf synsets, got {len(words)}')
    names = {w: s.split(',')[0].strip() for w, s in words.items()}
    return sorted(names), names, by_id


def _hf_classes(path):
    import ast
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'IMAGENET2012_CLASSES' for t in node.targets):
            value = node.value.args[0] if isinstance(node.value, ast.Call) else node.value     # OrderedDict({...})
            words = ast.literal_eval(value)
            return {w: HF_RENAMED.get(w, s) for w, s in words.items()}
    raise ValueError(f'{path}: no IMAGENET2012_CLASSES')


def base(dcfg, dataset):
    """'imagenet' or 'coco': whose prompt rules and real set a dataset uses ('split' for the fixed splits)."""
    c = dcfg[dataset]
    return 'split' if c.get('split') else c.get('base', dataset)


def prompts(dcfg, dataset, n_prompts=None, samples=None):
    """List of dict(id, index, prompt, seeds[, wnid])."""
    c = dcfg[dataset]
    samples = samples or c.get('samples') or dcfg.samples_per_prompt
    if c.get('split'):
        from .splits import split
        rows, seeds = split(c.split)
        if samples > len(seeds):
            raise ValueError(f'{dataset} has {len(seeds)} seeds')
        rows = [dict(r, index=i, seeds=seeds[:samples]) for i, r in enumerate(rows)]
        return _subset(rows, n_prompts) if n_prompts else rows
    if samples > SEED_STRIDE:
        raise ValueError(f'at most {SEED_STRIDE} samples per prompt')
    if base(dcfg, dataset) == 'imagenet':
        b = dcfg.imagenet
        wnids, names, _ = imagenet_classes(b)
        rows = [dict(id=f'in{i:04d}_{w}', prompt=b.template.format(names[w]), wnid=w) for i, w in enumerate(wnids)]
    elif base(dcfg, dataset) == 'coco':
        rows = _read_captions(Path(c.prompts))
    elif base(dcfg, dataset) == 'vbench':
        rows = _read_vbench(c)
    else:
        raise KeyError(dataset)
    for i, r in enumerate(rows):
        r.update(index=i, seeds=[c.seed_base+SEED_STRIDE*i+j for j in range(samples)])
    if c.get('n_classes'):
        rows = _subset(rows, c.n_classes)
    if c.get('n_subset'):
        rows = _subset(rows, c.n_subset)
    if n_prompts:
        rows = _subset(rows, n_prompts)
    return rows


def _read_vbench(c):
    lines = lambda p: [line.strip() for line in Path(p).read_text().splitlines() if line.strip()]
    prompts = lines(c.prompts)
    short = lines(c.short_prompts) if c.get('short_prompts') else None
    if short is not None and len(short) != len(prompts):
        raise ValueError(f'{c.prompts} and {c.short_prompts} differ in length')
    cats = list(c.get('categories') or [])
    if cats and len(prompts) != len(cats)*c.per_category:
        raise ValueError(f'{c.prompts}: expected {len(cats)} x {c.per_category} prompts')
    rows = []
    for i, p in enumerate(prompts):
        rows.append(dict(id=f'{c.id_prefix}{i:03d}'+(f'_{cats[i//c.per_category]}' if cats else ''), prompt=p))
        if short is not None:
            rows[-1]['short'] = short[i]
    return rows


def _subset(rows, n):
    """First n of a fixed shuffle, in the original order."""
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    return [rows[i] for i in sorted(order[:n])]


def _read_captions(path):
    if not path.exists():
        raise FileNotFoundError(f'COCO prompt file {path} not found (set data.coco.prompts)')
    if path.suffix == '.txt':
        items = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == '.jsonl':
        items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        items = json.loads(path.read_text())
        if isinstance(items, dict):                                # COCO caption file
            items = items['annotations']
    rows = []
    for i, it in enumerate(items):
        if isinstance(it, str):
            rows.append(dict(id=f'coco{i:04d}', prompt=it.strip()))
        else:
            rows.append(dict(id=f"coco_{it.get('id', i)}", prompt=it.get('caption', it.get('prompt')).strip()))
            if 'image_id' in it:
                rows[-1]['image_id'] = it['image_id']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate prompt ids in '+str(path))
    return rows


def real_images(dcfg, dataset):
    dataset = base(dcfg, dataset)
    if dataset == 'split':
        raise ValueError('the fixed splits have no real set')
    c = dcfg[dataset]
    if dataset == 'vbench':
        raise ValueError('VBench has no real video set')
    if dataset == 'coco':
        return sorted(Path(c.real_dir).glob('*.jpg'))
    wnids, _, by_id = imagenet_classes(c)
    if by_id is None:                                           # val_dir/<wnid>/ILSVRC2012_val_*.JPEG
        return [p for w in wnids for p in sorted((Path(c.val_dir)/w).glob('*.JPEG'))[:c.real_per_class]]
    truth = (Path(c.devkit)/'data'/'ILSVRC2012_validation_ground_truth.txt').read_text().split()
    per = {w: [] for w in wnids}
    for i, cls in enumerate(truth, start=1):                    # ground truth line i <-> ILSVRC2012_val_%08d.JPEG
        per[by_id[int(cls)]].append(Path(c.val_dir)/f'ILSVRC2012_val_{i:08d}.JPEG')
    return [p for w in wnids for p in sorted(per[w])[:c.real_per_class]]


def calibration_prompts(dcfg):
    path = dcfg.calibration.prompts
    if not path:
        raise ValueError('data.calibration.prompts is not set (the V-edit calibration prompts are still to be decided)')
    from .config import resolve
    return [r['prompt'] for r in _read_captions(resolve(path))]
