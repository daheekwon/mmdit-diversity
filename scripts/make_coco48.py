"""Write splits/coco48.json: 48 COCO val2017 images outside the COCO test and tune prompt files, one caption each.

  python scripts/make_coco48.py --captions <captions_val2017.json>

Images: sorted val2017 image ids, shuffled with random.Random(48); the first 48 that are in neither data.coco.prompts
nor data.coco_tune.prompts. Caption: one of the image's captions, chosen with random.Random(image id) from its captions
sorted by annotation id. Both prompt files must exist (otherwise the split could overlap the test set). An existing
splits/coco48.json is not overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import config, data, splits                                          # noqa: E402

N = 48


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--captions', required=True, help='COCO captions_val2017.json (5,000 images)')
    x = a.parse_args()
    if splits.COCO48.exists():
        raise SystemExit(f'{splits.COCO48} exists; remove it by hand to make a new split')
    d = config.load('sd35').data
    exclude = {}
    for name in ('coco', 'coco_tune'):
        rows = data.prompts(d, name)
        if not all('image_id' in r for r in rows):
            raise SystemExit(f'{d[name].prompts}: rows without image_id, cannot check overlap')
        exclude[name] = {r['image_id'] for r in rows}
    path = Path(x.captions)
    caps = json.loads(path.read_text())
    by_image = {}
    for an in caps['annotations']:
        by_image.setdefault(an['image_id'], []).append(an)
    images = sorted(by_image)
    if len(images) != 5000:
        raise SystemExit(f'{path}: {len(images)} images, expected val2017 (5,000)')
    random.Random(N).shuffle(images)
    banned = exclude['coco'] | exclude['coco_tune']
    chosen = [i for i in images if i not in banned][:N]
    rows = []
    for k, i in enumerate(chosen):
        an = random.Random(i).choice(sorted(by_image[i], key=lambda r: r['id']))
        rows.append(dict(id=f'coco_{k:02d}', prompt=an['caption'].strip(), image_id=i, caption_id=an['id']))
    meta = dict(source=str(path), source_md5=hashlib.md5(path.read_bytes()).hexdigest(), rule=__doc__.split('\n\n')[1],
                excluded={k: dict(file=str(d[k].prompts), images=len(v)) for k, v in exclude.items()})
    splits.COCO48.parent.mkdir(exist_ok=True)
    splits.COCO48.write_text(json.dumps(dict(meta=meta, prompts=rows), indent=1, ensure_ascii=False)+'\n')
    print(f'wrote {splits.COCO48}: {len(rows)} prompts, {len(banned)} test / tune images excluded')


if __name__ == '__main__':
    main()
