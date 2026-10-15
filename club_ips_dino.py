"""
Club-IPS with DINO prototypes (faithful to TF-SSD's IPS)
========================================================
Co-saliency over clubbed embeddings fails with SAM2 features alone, because SAM
embeddings are appearance-based — backgrounds look alike across a group and win
the consensus (exactly the issue TF-SSD reports). The fix, as in the paper, is to
describe each club with a DINO prototype (semantic): two differently-coloured
clocks get similar DINO features, while backgrounds differ.

Pipeline:
  for each club (region) in each image:
     prototype = DINO( image masked to that club )   # CLS token, semantic
  then run IPS: a club is co-salient if it has a strong best-match in EVERY
  other image of the group.

Requires: torch.hub DINOv2 (downloads once).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from club_ips import club_ips   # reuse the IPS scoring on prototypes

_DINO = None
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load_dino(device):
    global _DINO
    if _DINO is None:
        _DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        _DINO.eval().to(device)
    return _DINO


def dino_prototype(img, mask, device, crop=224):
    """
    DINO CLS embedding of the image masked to `mask`, cropped to the mask's bbox.
    img: (H,W,3) uint8;  mask: (H,W) bool. Returns L2-normalised (384,) vector.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    region = (img * mask[..., None]).astype(np.uint8)[y0:y1, x0:x1]   # zero out background
    t = torch.from_numpy(region).permute(2, 0, 1).float()[None] / 255.0
    t = F.interpolate(t, size=(crop, crop), mode="bilinear", align_corners=False)
    t = (t - _MEAN) / _STD
    with torch.inference_mode():
        v = _load_dino(device)(t.to(device)).squeeze(0).cpu().numpy()   # (384,) CLS
    return v / (np.linalg.norm(v) + 1e-8)


def add_dino_prototypes(group, device="mps", min_frac=0.002):
    """
    For each image dict in `group` (must have 'label_map' + 'meta'->'file'),
    compute a DINO prototype per club and store it under 'club_dino'.
    Tiny clubs (< min_frac of image) are skipped.
    """
    for g in group:
        img = np.array(Image.open(g["meta"]["file"]).convert("RGB"))
        H, W = img.shape[:2]; area = H * W
        lm = g["label_map"]
        protos = {}
        for c in np.unique(lm):
            m = (lm == c)
            if m.sum() < min_frac * area:
                continue
            p = dino_prototype(img, m, device)
            if p is not None:
                protos[int(c)] = p.astype(np.float32)
        g["club_dino"] = protos
    return group


def club_ips_dino(group, device="mps"):
    """Convenience: compute DINO prototypes then run IPS on them."""
    add_dino_prototypes(group, device)
    return club_ips(group, key="club_dino")
