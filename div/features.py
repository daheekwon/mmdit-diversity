"""Image and text features for the metrics, cached per folder as features/<name>.npz (rows follow the file list).

inception  2048-d pool3 of the TF-ported Inception-v3 as in clean-fid (PIL bicubic resize to 299): FID, PRDC
clip       CLIP ViT-B/32 image embedding: Vendi (as in the DAVE paper) and CLIP score
dino       DINOv2-small CLS: Vendi with the DINO kernel (this project's main diversity metric)
clip_l14   CLIP ViT-L/14 image embedding: LAION aesthetic predictor
pick       PickScore image embedding
All embeddings except inception are L2-normalised. Real images are resized (shorter side, bicubic) and centre-cropped
to the generation resolution first, so real and generated images reach every extractor at the same size.
"""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HF = dict(clip='openai/clip-vit-base-patch32', clip_l14='openai/clip-vit-large-patch14', dino='facebook/dinov2-small',
          pick='yuvalkirstain/PickScore_v1')
PICK_PROCESSOR = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
WEIGHTS = Path(__file__).resolve().parents[1]/'weights'      # inception-2015-12-05.pt is downloaded here


def load_image(path, size=None):
    im = Image.open(path).convert('RGB')
    if size and im.size != (size, size):
        s = size/min(im.size)
        im = im.resize((max(size, round(im.width*s)), max(size, round(im.height*s))), Image.BICUBIC)
        left, top = (im.width-size)//2, (im.height-size)//2
        im = im.crop((left, top, left+size, top+size))
    return im


class Inception:
    """clean-fid's 'clean' mode on one device (clean-fid itself wraps the network in DataParallel over every GPU)."""

    def __init__(self, device):
        from cleanfid.inception_torchscript import InceptionV3W
        from cleanfid.resize import make_resizer
        self.device = device
        self.net = InceptionV3W(str(WEIGHTS), download=True, resize_inside=False).to(device).eval()
        self.resize = make_resizer('PIL', False, 'bicubic', (299, 299))

    @torch.no_grad()
    def __call__(self, ims):
        x = torch.stack([torch.from_numpy(self.resize(np.array(im))).permute(2, 0, 1) for im in ims]).to(self.device)
        return self.net(x).double().cpu().numpy()


class Embedder:
    """CLIP / DINOv2 / PickScore through transformers (local Hugging Face cache)."""

    def __init__(self, name, device):
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        self.name, self.device = name, device
        repo = HF[name]
        self.net = AutoModel.from_pretrained(repo, local_files_only=True).to(device).eval()
        if name == 'dino':
            self.proc = AutoImageProcessor.from_pretrained(repo, local_files_only=True)
        else:
            self.proc = AutoProcessor.from_pretrained(PICK_PROCESSOR if name == 'pick' else repo, local_files_only=True)

    @torch.no_grad()
    def __call__(self, ims):
        if self.name == 'dino':                       # whole image at 224 (no crop), as value_diversity/study_eval.py
            x = self.proc(images=ims, return_tensors='pt', size={'shortest_edge': 224}, do_center_crop=False).to(self.device)
            e = self.net(**x).last_hidden_state[:, 0]
        else:
            e = self.net.get_image_features(**self.proc(images=ims, return_tensors='pt').to(self.device))
        return F.normalize(e.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def text(self, texts):
        x = self.proc(text=list(texts), padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)
        return F.normalize(self.net.get_text_features(**x).float(), dim=-1).cpu().numpy()

    @property
    def logit_scale(self):
        return self.net.logit_scale.exp().item()


def extractor(name, device):
    return Inception(device) if name == 'inception' else Embedder(name, device)


def cached(folder, name, files, device, size=None, batch=50, model=None):
    """Features of `files` (paths under `folder`), cached in folder/features/<name>.npz."""
    folder = Path(folder)
    path = folder/'features'/f'{name}.npz'
    keys = [str(Path(f).relative_to(folder)) if Path(f).is_relative_to(folder) else str(f) for f in files]
    if path.exists():
        z = np.load(path)
        if list(z['files']) == keys:
            return z['x']
    net = model or extractor(name, device)
    x = np.concatenate([net([load_image(f, size) for f in files[i:i+batch]]) for i in range(0, len(files), batch)])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npz')            # atomic: runs in parallel share the real-set cache
    np.savez(tmp, x=x, files=np.array(keys))
    os.replace(tmp, path)
    return x
