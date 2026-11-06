"""E06 VQA and object size (PROTOCOL_ko.md §4), reusing value_diversity's scorers with E06's questions and nouns.

  python experiments/E06_boundary/eval_extra.py vqa sd35 --device cuda:0      # rare + comp prompts of e06
  python experiments/E06_boundary/eval_extra.py objsize sd35 --device cuda:0  # every e06 prompt

VQA reads a view folder <model>/e06/vqa_view/<pid> -> generation/<pid> (rare_* and comp_* only), so its csv lands in
<model>/e06/vqa_0of1.csv. Object size writes <model>/e06/objsize_0of1.csv. Both resume.
"""
import argparse
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                            # noqa: E402
sys.path.insert(0, str(C.REPO))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('kind', choices=['vqa', 'objsize'])
    a.add_argument('model', choices=['sd35', 'flux'])
    a.add_argument('--device', default='cuda:0')
    x = a.parse_args()
    gen = C.gen_root(x.model, 'e06')
    rows = {r['id']: r for r in C.prompts('e06')}
    if x.kind == 'vqa':
        view = gen.parent/'vqa_view'
        view.mkdir(exist_ok=True)
        for pid, r in rows.items():
            if r['group'] in ('rare', 'comp') and not (view/pid).exists():
                (view/pid).symlink_to(gen/pid, target_is_directory=True)
        import value_diversity.xmodel_vqa as X
        X.HARD = [dict(id=pid, prompt=rows[pid]['prompt'], checks=[(q, 'yes') for q in qs]) for pid, qs in C.CHECKS.items()]
        sys.argv = ['xmodel_vqa', str(view), '--device', x.device]
        X.main()
    else:
        import value_diversity.location_objsize as O
        O.OBJECTS.update(C.OBJECTS)
        sys.argv = ['location_objsize', str(gen), '--device', x.device]
        O.main()


if __name__ == '__main__':
    main()
