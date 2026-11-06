"""Tables of PROTOCOL_ko.md section 7e from the six runs' metrics (div/video_metrics.py 'flow': motion, dynamic).

  python experiments/E02_t2v_cogvideox_pilot/analyze_motion.py

Motion ratio of a run = geometric mean over prompts of (run motion / original motion), with a prompt bootstrap 95%
interval (log ratios resampled). Pre-set rule: motion loss is confirmed when the ratio < 0.8 and the upper end of the
interval of the difference from the original (scripts/compare.py, outputs/cogvideox/pilot/motion_compare.md) < 0.
Writes outputs/cogvideox/pilot/motion.md.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from div import config                                                        # noqa: E402
from div.analysis import bootstrap                                            # noqa: E402

RUNS = config.ROOT/'outputs'/'cogvideox'/'vbench_tune'
NAMES = {'original': '원본', 'pad_v': 'pad V만', 'ours': '우리 방법', 'v_edit': 'V편집만',
         'ours_time_mean': '우리 방법(시간 평균)', 'v_edit_time_mean': 'V편집만(시간 평균)'}
STILL = ('vd924', 'vd465', 'vd933', 'vd139')                                 # still-life prompts (section 7e)


def main():
    per = {t: pd.read_csv(RUNS/t/'metrics'/'per_prompt.csv', index_col='id') for t in NAMES}
    base = per['original']
    rows = []
    for t, name in NAMES.items():
        lr = np.log(per[t].motion/base.motion)
        lo, hi = np.exp(bootstrap(lr))
        d = per[t].motion-base.motion
        dlo, dhi = bootstrap(d)
        moving = [i for i in base.index if i not in STILL]
        rows.append({'조건': name, '움직임 평균': per[t].motion.mean(), '원본 대비 차이': d.mean(), '차이 95%': f'[{dlo:.2f}, {dhi:.2f}]',
                     '비율 (기하 평균)': np.exp(lr.mean()), '비율 95%': f'[{lo:.2f}, {hi:.2f}]',
                     '비율, 정물 제외 12개': np.exp(lr[moving].mean()), '움직이는 영상 비율': per[t].dynamic.mean()})
    lines = ['## 영상 움직임 (RAFT 상위 5% 광류 크기, 픽셀/프레임; prompt 16개)', '',
             pd.DataFrame(rows).round(3).to_markdown(index=False), '',
             '## prompt별 영상 움직임', '',
             pd.DataFrame({NAMES[t]: per[t].motion for t in NAMES}).round(2).to_markdown(), '',
             '## prompt별 움직이는 영상 비율 (VBench 규칙)', '',
             pd.DataFrame({NAMES[t]: per[t].dynamic for t in NAMES}).round(2).to_markdown(), '']
    text = '\n'.join(lines)
    (config.ROOT/'outputs'/'cogvideox'/'pilot'/'motion.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
