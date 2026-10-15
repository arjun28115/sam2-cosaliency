"""
Selection lab: deep SAM2 descriptors for finalists.
SAM2's neck output (image_embed, 256-d) is tuned for mask decoding. The Hiera
trunk's deeper stages (stage 3: 576-d @64x64, stage 4: 1152-d @32x32) are more
semantic. Same model, different layer -> still SAM2-only.

Usage: python3 lab_trunk.py <tag> [<tag> ...]
Writes data/coca/trunk_<tag>.npz with L2-normalised descriptors per finalist:
  t3m, t4m (masked crop), t3r, t4r (raw crop), n_m (neck, masked; sanity check).
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
import v_reencode as vr

D = "/Users/arjunsingla/sam2/data/coca"
pred = vr._predictor()
_cap = {}
pred.model.image_encoder.trunk.register_forward_hook(lambda m, i, o: _cap.__setitem__("t", o))


def _pool(F, cmask):
    """Area-weighted mean of feature map F (C,h,w) over a full-res crop mask."""
    C, h, w = F.shape
    wgt = np.array(Image.fromarray(cmask.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR),
                   np.float32) / 255.0
    if wgt.sum() < 0.5:
        wgt = np.ones((h, w), np.float32)
    v = (F * wgt[None]).reshape(C, -1).sum(1) / wgt.sum()
    return v / (np.linalg.norm(v) + 1e-8)


def encode(img, mask_full, masked, margin=0.15):
    ys, xs = np.where(mask_full)
    H, W = mask_full.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    y0, y1, x0, x1 = max(0, y0 - my), min(H, y1 + my), max(0, x0 - mx), min(W, x1 + mx)
    crop = img[y0:y1, x0:x1].copy(); cm = mask_full[y0:y1, x0:x1]
    if masked:
        crop[~cm] = crop.reshape(-1, 3).mean(0).astype(np.uint8)
    with torch.inference_mode():
        pred.set_image(crop)
    t = _cap["t"]
    return (_pool(t[2][0].float().cpu().numpy(), cm),
            _pool(t[3][0].float().cpu().numpy(), cm),
            _pool(pred._features["image_embed"][0].float().cpu().numpy(), cm))


for tag in sys.argv[1:]:
    out = Path(f"{D}/trunk_{tag}.npz")
    if out.exists():
        print(f"[trunk] {tag}: cached", flush=True); continue
    z = np.load(f"{D}/lab_{tag}.npz", allow_pickle=True)
    data = np.load(f"{D}/{tag}.npy", allow_pickle=True)
    M = len(z["img"]); res = {k: [] for k in ["t3m", "t4m", "n_m", "t3r", "t4r"]}
    cur = -1
    for i in range(M):
        n = int(z["img"][i])
        if n != cur:
            img = np.array(Image.open(str(z["files"][n])).convert("RGB"))
            lm = data[n]["label_map"]
            lmf = np.array(Image.fromarray(lm.astype(np.int32), mode="I")
                           .resize((img.shape[1], img.shape[0]), Image.NEAREST))
            cur = n
        mf = np.isin(lmf, z["members"][i])
        if not mf.any():
            for k in res: res[k].append(np.zeros(576 if "3" in k else 1152 if "4" in k else 256, np.float32))
            continue
        a3, a4, an = encode(img, mf, masked=True)
        b3, b4, _ = encode(img, mf, masked=False)
        res["t3m"].append(a3); res["t4m"].append(a4); res["n_m"].append(an)
        res["t3r"].append(b3); res["t4r"].append(b4)
    np.savez_compressed(out, **{k: np.stack(v).astype(np.float32) for k, v in res.items()})
    print(f"[trunk] {tag}: {M} finalists -> saved", flush=True)
