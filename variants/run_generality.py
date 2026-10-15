"""Run the FINAL method unchanged (K=10, q=0.3, lam=0.15) on another dataset.
Usage: python3 run_generality.py <name>     # name in {clover, dice}
One dataset per process (module caches are dataset-specific)."""
import sys
from pathlib import Path
sys.path.insert(0, "/Users/arjunsingla/sam2/variants")

name = sys.argv[1]
import v_reencode as vr
vr.CACHE_DIR = Path(f"/Users/arjunsingla/sam2/data/reencode_cache_{name}")
vr.CACHE_DIR.mkdir(exist_ok=True)

from harness import load, evaluate
from v_final import run
import importlib
gt = importlib.import_module(f"gt_{name}")

data = load(f"/Users/arjunsingla/sam2/data/{name}_color.npy")
picks = run(data, K=10, q=0.3, lam=0.15)
hits, misses, det = evaluate(picks, data, boxes=gt.BOXES)
print(f"[GENERALITY {name}] {len(hits)}/{len(data)} miss={misses}", flush=True)
for d in det:
    print("  ", d, flush=True)

# render
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
N = len(data)
rows = (N + 3) // 4
fig, ax = plt.subplots(rows, 4, figsize=(16, 4 * rows))
for n in range(rows * 4):
    a = ax[n // 4, n % 4]
    a.axis("off")
    if n >= N:
        continue
    img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
    mem, m = picks[n]
    m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
        (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
    out = (img * 0.32).astype(np.uint8); out[m] = img[m]
    b = binary_dilation(find_boundaries(m, mode="outer"), iterations=3)
    hit = n in hits
    out[b] = (0, 255, 60) if hit else (255, 40, 40)
    a.imshow(out)
    a.set_title(("OK " if hit else "MISS ") + f"{n:02d} {mem}", fontsize=9,
                color=("green" if hit else "red"))
fig.suptitle(f"FINAL method on {name} — {len(hits)}/{N}", fontsize=14)
plt.tight_layout()
plt.savefig(f"/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency/GENERAL_{name}.png",
            dpi=100, bbox_inches="tight")
print("saved", f"GENERAL_{name}.png", flush=True)
