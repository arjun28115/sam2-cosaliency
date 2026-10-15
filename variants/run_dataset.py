"""
End-to-end runner: image folder -> final co-saliency results.
=============================================================
  1. pipeline_final.py   (SNIC -> prompts -> proposals -> NMS -> subtraction
                          -> super-embeddings -> clubbed super-embeddings)
  2. v_final.run         (min-Chamfer + evidence + SAM2 crop re-encode
                          + soft-AND consensus clique + prune/expand)
  3. multi_instance      (winner-calibrated top-3 bar for extra instances)
  4. refine_masks        (SAM2 decoder mask refinement)
  5. render              (green = winner, orange = extra instance)

Usage:  python3 run_dataset.py "/path/to/image folder" [tag]
HD/4K safe: pipeline logic runs at WORK_MAXSIDE=1280; crop re-encoding and
mask refinement use the ORIGINAL full-resolution files.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")

img_dir = Path(sys.argv[1])
tag = sys.argv[2] if len(sys.argv) > 2 else img_dir.name.replace(" ", "_")
data_npy = Path(f"/Users/arjunsingla/sam2/data/{tag}.npy")
out_dir = Path(f"/tmp/{tag}_run")

# 1. pipeline (skip if data already converted)
if not data_npy.exists():
    subprocess.run([sys.executable, "/Users/arjunsingla/sam2/pipeline_final.py",
                    "--image_dir", str(img_dir), "--output_dir", str(out_dir)], check=True)
    store = np.load(next(out_dir.glob("*_final.npy")), allow_pickle=True).item()
    data = [store[k] for k in sorted(store.keys())]
    np.save(data_npy, np.array(data, dtype=object))
    print(f"converted -> {data_npy}", flush=True)

import v_reencode as vr
vr.CACHE_DIR = Path(f"/Users/arjunsingla/sam2/data/reencode_cache_{tag}")
vr.CACHE_DIR.mkdir(exist_ok=True)

from harness import load
from v_final import run
from multi_instance import multi_instance
from refine_masks import refine_mask, full_res_mask

data = load(str(data_npy))
N = len(data)
print(f"{N} images loaded", flush=True)

# 2. selection
picks = run(data, K=10, q=0.3, lam=0.15)
print("selection done:", [m for m, _ in picks], flush=True)

# 3. extra instances
extras = multi_instance(data, picks)

# 4+5. refinement + render
pred = vr._predictor()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation, binary_fill_holes

rows = (N + 3) // 4
fig, ax = plt.subplots(rows, 4, figsize=(16, 4 * rows))
ax = np.atleast_2d(ax)
for n in range(rows * 4):
    a = ax[n // 4, n % 4]
    a.axis("off")
    if n >= N:
        continue
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
    a.imshow(out)
    a.set_title(f"{n:02d}" + (f" +{len(extras[n])}" if extras[n] else ""), fontsize=10)
    print(f"refined {n:02d}", flush=True)
fig.suptitle(f"SAM2-only co-saliency — {tag}", fontsize=14)
plt.tight_layout()
out_png = f"/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency/RESULT_{tag}.png"
plt.savefig(out_png, dpi=105, bbox_inches="tight")
print("saved", out_png, flush=True)
