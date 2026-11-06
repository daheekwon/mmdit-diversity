"""GPU check of the optical-flow motion metric (PROTOCOL_ko.md section 7e) before the six runs.

  python experiments/E02_t2v_cogvideox_pilot/check_flow.py --gpu 0 [--package <folder holding div/>]

1. Known motion: frame 0 of an original video and the same frame moved by (dx, dy) = (5, 3) pixels (edge rows and
   columns repeated). Expected: flow about (5, 3) away from the border, pair score about sqrt(34) = 5.83.
2. A frame paired with itself: expected pair score about 0.
3. One real video (original vd707, first seed): its 48 pair scores and the time per video.
Results: outputs/cogvideox/pilot/flow_check.json.
"""
import argparse
import json
from pathlib import Path
import sys
import time
HERE = Path(__file__).resolve().parent
a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument('--gpu', type=int, default=0)
a.add_argument('--package', default=str(HERE.parents[1]))
x = a.parse_args()
sys.path.insert(0, x.package)
import numpy as np                                                            # noqa: E402
import torch                                                                  # noqa: E402
import torch.nn.functional as F                                               # noqa: E402
from div import video_metrics as vm                                           # noqa: E402

ROOT = HERE.parents[1]
RUN = ROOT/'outputs'/'cogvideox'/'vbench_tune'/'original'


def main():
    import os
    os.environ['TORCH_HOME'] = str(vm.ANAL/'model_cache'/'torch_eval')
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    dev = f'cuda:{x.gpu}'
    w = Raft_Large_Weights.C_T_V1
    net, tf = raft_large(weights=w, progress=False).eval().to(dev), w.transforms()
    row = {r['id']: r for r in vm.run_rows(RUN)}['vd707']
    v = torch.from_numpy(vm.read_video(row['files'][0])).permute(0, 3, 1, 2)
    f0 = v[:1]
    dx, dy = 5, 3
    moved = F.pad(f0.float(), (dx, 0, dy, 0), mode='replicate')[..., :f0.shape[-2], :f0.shape[-1]].round().to(torch.uint8)
    res = {}
    with torch.no_grad():
        for name, b in (('shift', moved), ('same', f0)):
            a1, b1 = tf(f0, b)
            flow = net(a1.to(dev), b1.to(dev), num_flow_updates=vm.FLOW_UPDATES)[-1][0].float()
            inner = flow[:, 20:-20, 20:-20]
            res[name] = dict(pair_score=float(vm.top_mean(flow.norm(dim=0)[None])[0]),
                             median_dx=float(inner[0].median()), median_dy=float(inner[1].median()))
    t0 = time.monotonic()
    fl = vm.flow_frames(Path('/nonexistent'), [], dev, compute=False)         # cache path unused: compute directly below
    assert fl is None
    with torch.no_grad():
        s = []
        for i in range(0, vm.N_FRAMES-1, vm.FLOW_BATCH):
            j = min(i+vm.FLOW_BATCH, vm.N_FRAMES-1)
            a1, b1 = tf(v[i:j], v[i+1:j+1])
            s.append(vm.top_mean(net(a1.to(dev), b1.to(dev), num_flow_updates=vm.FLOW_UPDATES)[-1].float().norm(dim=1)).cpu())
    s = torch.cat(s).numpy()
    res['real'] = dict(video=str(row['files'][0].relative_to(ROOT)), seconds=round(time.monotonic()-t0, 1),
                       motion=float(s.mean()), dynamic=vm.vbench_dynamic(s, min(v.shape[-2:])),
                       pair_scores=[round(float(c), 2) for c in s])
    res['expected_shift_score'] = float(np.hypot(dx, dy))
    (ROOT/'outputs'/'cogvideox'/'pilot'/'flow_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
