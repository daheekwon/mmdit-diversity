"""Evaluate runs (see div/metrics.py for the metrics).

  python scripts/evaluate.py outputs/sd35/coco/ours outputs/sd35/coco/original
  python scripts/evaluate.py outputs/flux/imagenet/dave --metrics vendi_dino clip_score

Writes <run>/metrics/{per_image.csv, per_prompt.csv, summary.json}. Features are cached in <run>/features/ and the real
sets' Inception features in outputs/real/<dataset>_<resolution>/.
"""
import argparse
import json
from pathlib import Path
import sys
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import config, metrics                                               # noqa: E402


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('runs', nargs='+')
    a.add_argument('--metrics', nargs='*', default=list(metrics.PER_IMAGE+metrics.PER_PROMPT+metrics.SET),
                   choices=metrics.PER_IMAGE+metrics.PER_PROMPT+metrics.SET)
    a.add_argument('--device', default='cuda:0')
    x = a.parse_args()
    for run in x.runs:
        cfg = OmegaConf.create(json.loads((Path(run)/'run.json').read_text())['config'])
        cfg.data = OmegaConf.load(config.ROOT/'configs'/'data.yaml')          # current dataset paths
        print(json.dumps(metrics.evaluate(run, cfg, x.device, x.metrics), indent=1), flush=True)


if __name__ == '__main__':
    main()
