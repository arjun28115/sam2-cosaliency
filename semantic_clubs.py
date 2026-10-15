"""
Dual-stream Stage 7: attach a SigLIP patch-feature semantic embedding to each club.
Stages 0-6 (SAM2 latent-SNIC + clubbing) are unchanged — this only adds a second,
semantic descriptor per club, drawn from SigLIP's dense patch grid.

  1. resize image to 224, forward SigLIP vision encoder, take patch tokens (14x14x768)
  2. bilinear-upsample the 14x14 grid to the club label_map resolution
  3. club embedding = mean of the semantic tensor over the club mask, L2-normalised
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SIG = None
_GRID = None  # patches per side


def _load(device="mps", name="google/siglip-base-patch16-224"):
    global _SIG, _GRID
    if _SIG is None:
        from transformers import SiglipVisionModel
        _SIG = SiglipVisionModel.from_pretrained(name).eval().to(device)
        ps = _SIG.config.patch_size; im = _SIG.config.image_size
        _GRID = im // ps                                  # 224/16 = 14
    return _SIG


def semantic_feature_map(img, device="mps"):
    """Image (H,W,3) -> dense semantic tensor (grid, grid, 768) from SigLIP patches."""
    m = _load(device)
    t = torch.from_numpy(np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))) \
             .permute(2, 0, 1).float()[None] / 255.0
    t = (t - 0.5) / 0.5                                    # SigLIP normalisation [-1,1]
    with torch.inference_mode():
        tok = m(pixel_values=t.to(device)).last_hidden_state.squeeze(0)   # (196, 768)
    D = tok.shape[-1]
    return tok.reshape(_GRID, _GRID, D).cpu().float().numpy()


def add_semantic_embeddings(group, device="mps", min_frac=0.0):
    """For each image dict (needs 'label_map' + 'meta'->'file') add 'club_semantic_embeddings'."""
    from scipy.ndimage import zoom
    for g in group:
        img = np.array(Image.open(g["meta"]["file"]).convert("RGB"))
        lm = g["label_map"]; H, W = lm.shape
        grid = semantic_feature_map(img, device)                       # (14,14,768)
        Vsem = zoom(grid, (H / grid.shape[0], W / grid.shape[1], 1), order=1)  # (H,W,768) bilinear
        emb = {}
        for c in np.unique(lm):
            v = Vsem[lm == c].mean(0)                                  # mean over club pixels
            emb[int(c)] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
        g["club_semantic_embeddings"] = emb
    return group
