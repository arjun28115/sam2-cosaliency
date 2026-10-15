"""Render each image with numbered club overlays for GT annotation."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries

data = np.load(sys.argv[1], allow_pickle=True)
out = sys.argv[2]
N = len(data)
fig, ax = plt.subplots(4, 4, figsize=(22, 20))
for n in range(N):
    a = ax[n // 4, n % 4]
    img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
    lm = data[n]["label_map"]
    lm_big = np.array(Image.fromarray(lm.astype(np.int32), mode="I").resize(
        (img.shape[1], img.shape[0]), Image.NEAREST))
    vis = (mark_boundaries(img, lm_big, color=(0, 1, 1), mode="thick") * 255).astype(np.uint8)
    a.imshow(vis)
    H, W = lm_big.shape
    for cid in np.unique(lm_big):
        m = lm_big == cid
        if m.sum() < 0.001 * m.size:
            continue
        ys, xs = np.where(m)
        a.text(xs.mean(), ys.mean(), str(int(cid)), color="yellow", fontsize=11,
               ha="center", va="center",
               bbox=dict(facecolor="black", alpha=0.6, pad=1))
    a.set_title(f"{n:02d}  ({data[n]['n_clubs']} clubs)", fontsize=11)
    a.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=100, bbox_inches="tight")
print("saved", out)
