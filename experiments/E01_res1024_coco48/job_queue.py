"""Runs value_diversity/combo_queue.py on a queue folder under another name, so that waiters that look for the shared
queue script by command line do not mistake this queue for theirs.  python experiments/E01_res1024_coco48/job_queue.py <qdir>"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from value_diversity.combo_queue import main                                  # noqa: E402

if __name__ == '__main__':
    main(sys.argv[1])
