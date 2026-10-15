"""
Multi-scale super-embeddings — same pipeline flow, richer dense features.
=========================================================================
pipeline_final.py uses ONLY the 64x64 image_embed (each cell ~20px at
WORK_MAXSIDE=1280): tiny objects are averaged away before pooling. SAM2's
encoder also emits high_res_feats at 256x256x32 and 128x128x64 — 16x finer
spatial detail. This wrapper monkeypatches the THREE places the dense feature
map is built (SNIC input, superpixel pooling, club pixel-reference) to use a
fused multi-scale map:

    F(x) = [ w_ie * n(image_embed) | w_hr * n(hr_128) | w_hr * n(hr_256) ]

(n = per-pixel L2; concat -> 256+64+32 = 352-d). Everything downstream
(SNIC -> prompts -> proposals -> NMS -> subtraction -> super-embeddings ->
clubbing -> co-saliency) is unchanged and dimension-agnostic.

Usage: python3 pipeline_ms.py --image_dir ... --output_dir ...
"""
import sys
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, "/Users/arjunsingla/sam2")
import pipeline_final as pf

W_IE = 1.0     # semantic 256-d stream
W_HR = 0.5     # each high-res structural stream


def _nrm(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def _fused_map(predictor, H, W):
    """(H, W, 352) multi-scale dense feature map."""
    ie = predictor._features["image_embed"].squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    hrs = [t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
           for t in predictor._features["high_res_feats"]]      # [(256,256,32), (128,128,64)]
    parts = [W_IE * _nrm(zoom(ie, (H / ie.shape[0], W / ie.shape[1], 1), order=1))]
    for hr in sorted(hrs, key=lambda a: -a.shape[0]):           # finest first
        parts.append(W_HR * _nrm(zoom(hr, (H / hr.shape[0], W / hr.shape[1], 1), order=1)))
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ── patch 1: SNIC runs on the fused map (at SNIC_TARGET resolution) ───────────
def snic_superpixels_ms(predictor, H, W):
    t = pf.SNIC_TARGET
    fm = _fused_map(predictor, t, t)                       # same square grid as original
    seg = pf.snic_latent(fm, pf.n_superpixels(H, W), m=pf.SNIC_M)
    return zoom(seg, (H / t, W / t), order=0).astype(np.int32)


# ── patch 2: dense map used for pooling (dimension-agnostic) ──────────────────
def _image_embed_upsampled_ms(predictor, H, W):
    return _fused_map(predictor, H, W)


def super_embeddings_constrained_ms(predictor, segs, prior):
    H, W = segs.shape
    embed_up = _image_embed_upsampled_ms(predictor, H, W)
    D = embed_up.shape[-1]
    ids = np.unique(segs)
    mat = np.zeros((len(ids), D), dtype=np.float32)
    sizes = np.zeros(len(ids), dtype=np.int64)
    sp_to_club = {}
    for i, sid in enumerate(ids):
        m = (segs == sid)
        v = embed_up[m].mean(0)
        mat[i] = v / (np.linalg.norm(v) + 1e-8)
        sizes[i] = int(m.sum())
        vals, cnts = np.unique(prior[m], return_counts=True)
        sp_to_club[int(sid)] = int(vals[cnts.argmax()])
    return ids, mat, sizes, sp_to_club


pf.snic_superpixels = snic_superpixels_ms
pf._image_embed_upsampled = _image_embed_upsampled_ms
pf.super_embeddings_constrained = super_embeddings_constrained_ms

if __name__ == "__main__":
    pf.main()
