"""CPU checks of div/video_metrics.py against brute-force references; one end-to-end run on two tiny mp4 videos
(DINOv2-small on the CPU; MUSIQ and the judge are not run here)."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from div import video_metrics as vm                                           # noqa: E402
from div.metrics import vendi                                                 # noqa: E402

rng = np.random.default_rng(0)


def unit(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def test_subject_consistency_brute_force():
    F = unit(rng.normal(size=(49, 16)))
    ref = np.mean([(F[i]@F[i-1]+F[i]@F[0])/2 for i in range(1, 49)])
    assert np.isclose(vm.subject_consistency(F), ref)


def test_flicker_brute_force():
    v = rng.integers(0, 256, (5, 4, 6, 3), dtype=np.uint8)
    ref = np.mean([np.abs(v[i].astype(float)-v[i-1].astype(float)).mean() for i in range(1, 5)])
    assert np.isclose(vm.flicker_mae(v), ref)


def test_video_feature_and_vendi():
    F = unit(rng.normal(size=(3, 49, 16)))
    ref = [unit(np.mean([F[k, i] for i in vm.FRAMES9], 0)) for k in range(3)]
    assert np.allclose(vm.video_feature(F), ref)
    same = np.repeat(vm.video_feature(F)[:1], 4, 0)
    assert np.isclose(vendi(same), 1)
    assert np.isclose(vendi(np.eye(4)), 4)


def test_judge_rules():
    j = pd.DataFrame(dict(frame=vm.JUDGE_FRAMES, p_match=[.9, .2, .3, .4], p_defect=[.1, .1, .6, .1], p_style=[.1]*4))
    r = vm.judge_video(j)
    assert r['broken'] == 1 and r['mismatch'] == 1 and np.isclose(r['p_match'], .45)      # one defect frame; mean match < .5
    j2 = j.assign(p_defect=.1, p_match=.6)
    assert vm.judge_video(j2)['broken'] == 0 and vm.judge_video(j2)['mismatch'] == 0


def test_flow_rules():
    """top_mean against VBench's numpy sort; vbench_dynamic against VBench's check_move loop."""
    import torch
    rad = torch.rand(3, 20, 30)
    ref = [np.mean(np.abs(np.sort(-r.numpy().flatten()))[:int(600*.05)]) for r in rad]
    assert np.allclose(vm.top_mean(rad).numpy(), ref, atol=1e-6)

    def check_move(scores, short, count):              # VBench DynamicDegree.set_params + check_move
        thres, count_num, n = 6.0*(short/256.0), round(4*(count/16.0)), 0
        for s in scores[:-2]:
            if s > thres:
                n += 1
            if n >= count_num:
                return True
        return False
    for _ in range(200):
        s = rng.uniform(0, 20, 48)
        assert vm.vbench_dynamic(s, 480) == float(check_move(list(s), 480, 49))
    assert vm.vbench_dynamic(np.r_[np.full(12, 12.), np.zeros(36)], 480) == 1
    assert vm.vbench_dynamic(np.r_[np.zeros(36), np.full(12, 12.)], 480) == 0      # 2 of the 12 are the last two


def test_evaluate_end_to_end(tmp_path):
    from generate import save_video
    run = tmp_path/'run'
    rows = [dict(id='p0', index=0, prompt='a', short='a', seeds=[1, 2]), dict(id='p1', index=1, prompt='b', short='b', seeds=[3, 4])]
    (run/'videos').mkdir(parents=True)
    (run/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for r in rows:
        for s in r['seeds']:
            (run/'videos'/r['id']).mkdir(exist_ok=True)
            save_video(run/'videos'/r['id']/f'{s}.mp4', rng.integers(0, 256, (49, 32, 48, 3), dtype=np.uint8), 8)
    s = vm.evaluate(run, 'cpu', ('dino',))
    per = pd.read_csv(run/'metrics'/'per_prompt.csv', index_col='id')
    assert s['videos'] == 4 and list(per.index) == ['p0', 'p1']
    assert {'vendi_dino', 'vendi_dino_first', 'vendi_dino_frames', 'subject_consistency', 'flicker_mae'} <= set(per.columns)
    assert ((per.vendi_dino >= 1) & (per.vendi_dino <= 2)).all()
    z = np.load(run/'features'/'video_frames.npz')
    assert z['dino'].shape == (4, 49, 384)
    vid = pd.read_csv(run/'metrics'/'per_video.csv')
    v = vm.read_video(run/'videos'/'p0'/'1.mp4')
    assert np.isclose(vid.flicker_mae[0], vm.flicker_mae(v))
