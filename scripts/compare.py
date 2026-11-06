"""Compare evaluated runs with a reference run (paired by prompt; 95% bootstrap interval over prompts).

  python scripts/compare.py outputs/sd35/coco/original outputs/sd35/coco/ours outputs/sd35/coco/dave
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from div import analysis                                                      # noqa: E402


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('ref')
    a.add_argument('runs', nargs='+')
    a.add_argument('--metrics', nargs='*')
    x = a.parse_args()
    print(analysis.compare(x.ref, x.runs, x.metrics).to_markdown(index=False, floatfmt='.4g'))


if __name__ == '__main__':
    main()
