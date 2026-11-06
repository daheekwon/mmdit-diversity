"""Evaluate video runs (see div/video_metrics.py for the metrics).

  python scripts/evaluate_video.py outputs/cogvideox/vbench_tune/original outputs/cogvideox/vbench_tune/ours
  python scripts/evaluate_video.py outputs/cogvideox/vbench_tune/ours --metrics dino musiq     # judge later
  python scripts/evaluate_video.py outputs/cogvideox/vbench_tune/ours --metrics flow           # add one metric

Writes <run>/metrics/{per_video.csv, per_prompt.csv, summary.json}; then scripts/compare.py compares runs as for images.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import video_metrics                                                 # noqa: E402


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('runs', nargs='+')
    a.add_argument('--metrics', nargs='*', default=list(video_metrics.METRICS), choices=video_metrics.METRICS)
    a.add_argument('--device', default='cuda:0')
    x = a.parse_args()
    for run in x.runs:
        print(json.dumps(video_metrics.evaluate(run, x.device, x.metrics), indent=1), flush=True)


if __name__ == '__main__':
    main()
