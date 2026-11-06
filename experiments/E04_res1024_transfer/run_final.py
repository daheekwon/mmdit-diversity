"""Generate the final method (configs/<model>.yaml preset `ours`, nothing overridden but the resolution) on COCO 48 x seeds
68000-68007 with E01's generator, into outputs/e04_final/res<res>/<model>/coco48/generation/<pid>/<FINAL[model]>.

  python experiments/E04_res1024_transfer/run_final.py --model sd35 --res 1024 --shard 0/8 --gpu 0
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'E01_res1024_coco48'))
sys.path.insert(0, str(HERE))
import run_coco48 as R                                                        # noqa: E402
from common import FINAL, gen_root                                            # noqa: E402

R.MODES = {m: {FINAL[m]: ('ours', 8)} for m in FINAL}

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--root' not in argv:
        model, res = argv[argv.index('--model')+1], int(argv[argv.index('--res')+1])
        sys.argv += ['--root', str(gen_root(model, res))]
    R.main()
