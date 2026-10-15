import numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from cosal_union import union_cosaliency

data = np.load("/tmp/clock_grey.npy", allow_pickle=True); N = len(data)
OUT = "/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency"
tag = sys.argv[1] if len(sys.argv) > 1 else "union"
low = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
gam = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
fgp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
mu  = int(sys.argv[5]) if len(sys.argv) > 5 else 3

picks = union_cosaliency(data, low_frac=low, gamma=gam, fg_pow=fgp, max_union=mu)
fig, ax = plt.subplots(4, 4, figsize=(16, 15))
for n in range(N):
    a = ax[n // 4, n % 4]
    img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
    mem, m = picks[n]
    m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
        (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
    out = (img * 0.35).astype(np.uint8); out[m] = img[m]
    b = binary_dilation(find_boundaries(m, mode="outer"), iterations=2)
    out[b] = (0, 255, 255)
    a.imshow(out); a.set_title(f"{n:02d}  {mem}", fontsize=9); a.axis("off")
fig.suptitle(f"Union-candidate co-saliency  [{tag}]  low={low} g={gam} fg={fgp} K={mu}", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/{tag}.png", dpi=105, bbox_inches="tight"); plt.close()
print("saved", f"{OUT}/{tag}.png")
