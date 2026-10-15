import numpy as np, matplotlib, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from cosal_spectral import spectral_cosaliency, select

data = np.load("/tmp/clock_grey.npy", allow_pickle=True); N = len(data)
OUT = "/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency"
tag = sys.argv[1] if len(sys.argv) > 1 else "spectral"

score, img_of, club_of = spectral_cosaliency(data)
picks = select(score, img_of, club_of, data)
print("picks:", [c for c, _ in picks])

fig, ax = plt.subplots(4, 4, figsize=(16, 15))
for n in range(N):
    a = ax[n // 4, n % 4]
    img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
    cid, m = picks[n]
    m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
        (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
    out = (img * 0.35).astype(np.uint8); out[m] = img[m]
    b = binary_dilation(find_boundaries(m, mode="outer"), iterations=2)
    out[b] = (0, 255, 255)
    a.imshow(out); a.set_title(f"{n:02d}  club {cid}", fontsize=10); a.axis("off")
fig.suptitle(f"Spectral + RBD foreground co-saliency [{tag}]", fontsize=15)
plt.tight_layout(); plt.savefig(f"{OUT}/{tag}.png", dpi=105, bbox_inches="tight"); plt.close()
print("saved", f"{OUT}/{tag}.png")
