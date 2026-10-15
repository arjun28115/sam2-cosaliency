"""Diagnose the dice recall ceiling: at the PROPOSAL stage (before NMS and
subtraction), does ANY SAM2 mask isolate a die? If yes, the partition/NMS
loses them; if no, prompting/proposal-mode is the binding stage."""
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/arjunsingla/sam2")
import pipeline_final as pf
from gt_dice import BOXES

import os, torch
os.chdir("/Users/arjunsingla/sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
device = "mps" if torch.backends.mps.is_available() else "cpu"
pred = SAM2ImagePredictor(build_sam2(pf.CONFIG, str(pf.CHECKPOINT), device=device))

for n in [3, 10, 11]:                      # dice cup, tray of five, pink cloth
    p = f"/Users/arjunsingla/Desktop/Images/dice/{n:02d}.jpg"
    img_full = np.array(Image.open(p).convert("RGB"))
    H0, W0 = img_full.shape[:2]
    scale = pf.WORK_MAXSIDE / max(H0, W0)
    if scale < 1:
        img = np.array(Image.fromarray(img_full).resize(
            (int(W0 * scale), int(H0 * scale)), Image.LANCZOS))
    else:
        img = img_full
    H, W = img.shape[:2]
    with torch.inference_mode():
        pred.set_image(img)
    segs = pf.snic_superpixels(pred, H, W)
    pts = pf.superpixel_centroids(segs)
    proposals = pf.sam_point_proposals(pred, pts, H * W)
    masks = [m for m, s in proposals]
    print(f"img {n:02d}: {len(masks)} raw proposals (pre-NMS)", flush=True)

    for bi, (x0, y0, x1, y1) in enumerate(BOXES[n]):
        bm = np.zeros((H, W), bool)
        bm[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
        best = (0.0, None)
        for m in masks:
            m = m.astype(bool)
            inter = float((m & bm).sum())
            pur = inter / (m.sum() + 1e-8)
            cov = inter / (bm.sum() + 1e-8)
            v = min(pur / 0.6, cov / 0.25)
            if v > best[0]:
                best = (v, (round(pur, 2), round(cov, 2), round(m.mean(), 4)))
        print(f"   box{bi}: best proposal fit={best[0]:.2f} (purity,cover,area)={best[1]}", flush=True)
