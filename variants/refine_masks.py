"""
SAM2 mask refinement — turn coarse club-union selections into crisp masks.
==========================================================================
The pipeline's masks are partition regions (working-res, morphologically
cleaned, nearest-upsampled): correct object, ragged boundary. Fix: prompt SAM2
with the selected region itself on the FULL-RES image —
    mask prompt  (256x256 logits of the selection)
  + box prompt   (selection bbox)
  + anchor point (selection centroid, positive)
and take the decoder output that best matches the seed. SAM2's decoder gives
the pixel-accurate boundary; the co-saliency stage supplies WHAT to segment,
SAM2 supplies WHERE its edges are. Still SAM2-only.
"""
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_fill_holes, label as cc_label

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
import v_reencode as vr
from harness import load
from v_final import run
from multi_instance import multi_instance


def largest_component(mask):
    """Keep only the largest connected component (removes speckle islands)."""
    lab, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


def refine_mask(pred, img_full, seed_mask_full, margin=0.18):
    """ZOOM refinement: run the decoder on the object's CROP so SAM2's internal
    1024 budget covers just the object (HD boundary precision), with a padded
    box so parts the seed missed (bell tips, gear) can be recovered.
    seed (H0,W0) bool at full res -> crisp SAM2 mask (H0,W0) bool."""
    import torch
    H0, W0 = seed_mask_full.shape
    ys, xs = np.where(seed_mask_full)
    if len(ys) == 0:
        return seed_mask_full
    # crop window = seed bbox + margin (the decoder may grow into the margin)
    my = max(8, int((ys.max() - ys.min() + 1) * margin))
    mx = max(8, int((xs.max() - xs.min() + 1) * margin))
    cy0, cy1 = max(0, ys.min() - my), min(H0, ys.max() + 1 + my)
    cx0, cx1 = max(0, xs.min() - mx), min(W0, xs.max() + 1 + mx)
    crop = img_full[cy0:cy1, cx0:cx1]
    seed_c = seed_mask_full[cy0:cy1, cx0:cx1]
    ysc, xsc = np.where(seed_c)
    # padded prompt box inside the crop (lets the decoder extend past the seed)
    py0 = max(0.0, ysc.min() - my * 0.6); py1 = min(crop.shape[0] - 1.0, ysc.max() + my * 0.6)
    px0 = max(0.0, xsc.min() - mx * 0.6); px1 = min(crop.shape[1] - 1.0, xsc.max() + mx * 0.6)
    box = np.array([px0, py0, px1, py1], np.float32)
    cyc, cxc = float(ysc.mean()), float(xsc.mean())
    d2 = (ysc - cyc) ** 2 + (xsc - cxc) ** 2
    j = int(np.argmin(d2)); py, px = float(ysc[j]), float(xsc[j])
    m256 = np.array(Image.fromarray(seed_c.astype(np.uint8) * 255)
                    .resize((256, 256), Image.BILINEAR)).astype(np.float32) / 255.0
    logits = (m256 * 20.0 - 10.0)[None]                     # (1,256,256)
    with torch.inference_mode():
        pred.set_image(crop)
        masks, ious, _ = pred.predict(
            point_coords=np.array([[px, py]], np.float32),
            point_labels=np.array([1]),
            box=box[None],
            mask_input=logits,
            multimask_output=True)
    best, best_iou = None, -1.0
    for k in range(masks.shape[0]):
        m = masks[k] > 0
        inter = float((m & seed_c).sum())
        union = float((m | seed_c).sum()) + 1e-8
        if inter / union > best_iou:
            best_iou, best = inter / union, m
    out = np.zeros((H0, W0), bool)
    out[cy0:cy1, cx0:cx1] = binary_fill_holes(largest_component(best))
    return out


def full_res_mask(data, n, mask_lm):
    img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
    mf = np.array(Image.fromarray(mask_lm.astype(np.uint8) * 255)
                  .resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 127
    return img, mf


if __name__ == "__main__":
    vr.CACHE_DIR = Path("/Users/arjunsingla/sam2/data/reencode_cache")
    data = load()
    picks = run(data, K=10, q=0.3, lam=0.15)
    extras = multi_instance(data, picks)
    pred = vr._predictor()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation, binary_fill_holes

    fig, ax = plt.subplots(4, 4, figsize=(16, 15))
    for n in range(16):
        a = ax[n // 4, n % 4]
        img, seed = full_res_mask(data, n, picks[n][1])
        m = binary_fill_holes(refine_mask(pred, img, seed))
        out = (img * 0.3).astype(np.uint8); out[m] = img[m]
        b = binary_dilation(find_boundaries(m, mode="outer"), iterations=3)
        out[b] = (0, 255, 60)
        for (mem2, m2, sc) in extras[n]:
            _, seed2 = full_res_mask(data, n, m2)
            mm = binary_fill_holes(refine_mask(pred, img, seed2))
            out[mm] = img[mm]
            b2 = binary_dilation(find_boundaries(mm, mode="outer"), iterations=3)
            out[b2] = (255, 200, 0)
        a.imshow(out); a.set_title(f"{n:02d}", fontsize=10); a.axis("off")
        print(f"refined {n:02d}", flush=True)
    fig.suptitle("FINAL — co-saliency selections refined by SAM2 decoder (crisp masks)", fontsize=13)
    plt.tight_layout()
    plt.savefig("/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency/FINAL_refined.png",
                dpi=105, bbox_inches="tight")
    print("saved FINAL_refined.png", flush=True)
