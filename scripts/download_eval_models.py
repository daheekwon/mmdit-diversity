"""Download the evaluation models (2026-11-06; the earlier server's copies are not available here).

  source env.sh && python scripts/download_eval_models.py

Hugging Face repos go to $HF_HOME/hub (div/features.py loads them with local_files_only):
  openai/clip-vit-base-patch32       CLIP score, Vendi (CLIP kernel)
  openai/clip-vit-large-patch14      LAION aesthetic predictor input
  facebook/dinov2-small              Vendi (DINOv2 kernel)
  yuvalkirstain/PickScore_v1         PickScore (+ the processor of laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
  THUDM/ImageReward                  ImageReward-v1.0 (ImageReward.pt, med_config.json -> ~/.cache/ImageReward,
                                     where the image-reward package looks) + the bert-base-uncased tokenizer of its BLIP
Files under weights/:
  inception-2015-12-05.pt            clean-fid's TF-ported Inception (FID, PRDC)
  sac_logos_ava1_l14_linearMSE.pth   LAION aesthetic predictor v2 (improved-aesthetic-predictor, sac+logos+ava1-l14-linearMSE)
torchvision AlexNet (LPIPS 'alex' backbone) goes to $TORCH_HOME (default ~/.cache/torch).
Writes env/eval_models.json: the downloaded revisions and file sha256, so a later copy can be checked.
"""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT/'weights'
MODEL = ['*.json', '*.txt', 'model.safetensors']        # pytorch_model.bin where a repo has no safetensors
REPOS = {'openai/clip-vit-base-patch32': MODEL, 'openai/clip-vit-large-patch14': MODEL, 'facebook/dinov2-small': MODEL,
         'yuvalkirstain/PickScore_v1': MODEL, 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': ['*.json', '*.txt'],
         'bert-base-uncased': ['*.json', '*.txt']}
FILES = {'inception-2015-12-05.pt':
         'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',
         'sac_logos_ava1_l14_linearMSE.pth':
         'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth'}
IMAGE_REWARD = ['ImageReward.pt', 'med_config.json']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.environ.get('HF_HOME'):
        raise SystemExit('HF_HOME is not set: source env.sh first')
    out = dict(hf={}, files={}, image_reward={})
    for repo, patterns in REPOS.items():
        if patterns is MODEL and 'model.safetensors' not in HfApi().list_repo_files(repo):
            patterns = MODEL[:-1]+['pytorch_model.bin']          # e.g. openai/clip-vit-base-patch32 main: .bin only
        path = Path(snapshot_download(repo, allow_patterns=patterns))
        out['hf'][repo] = dict(revision=path.name, files=sorted(p.name for p in path.iterdir()))
        print(f'{repo} {path.name}', flush=True)
    WEIGHTS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dst = WEIGHTS/name
        if not dst.exists():
            tmp = dst.with_suffix('.part')
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)
        out['files'][name] = dict(url=url, sha256=sha256(dst), bytes=dst.stat().st_size)
        print(f'weights/{name} {out["files"][name]["sha256"][:12]}', flush=True)
    ir = Path.home()/'.cache'/'ImageReward'
    for name in IMAGE_REWARD:
        p = Path(hf_hub_download('THUDM/ImageReward', name, local_dir=ir))
        out['image_reward'][name] = dict(path=str(p), sha256=sha256(p))
        print(f'{p}', flush=True)
    import torchvision
    torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    out['alexnet'] = 'torchvision AlexNet_Weights.IMAGENET1K_V1'
    (ROOT/'env'/'eval_models.json').write_text(json.dumps(out, indent=1)+'\n')
    print('wrote env/eval_models.json')


if __name__ == '__main__':
    main()
