import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from cosal_perdim import per_dimension_consensus, _flatten

data = np.load("/tmp/clock_grey.npy", allow_pickle=True); N = len(data)
OUT = "/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency"

Z, score_pd, img_of, club_of, Sim = per_dimension_consensus(data)
P, img2, _ = _flatten(data)
C = P @ P.T
score_fl = np.zeros(len(P)); cnt = np.zeros(len(P))
for j in range(N):
    col = img2 == j; cmax = C[:, col].max(1); notj = img2 != j
    score_fl[notj] += cmax[notj]; cnt[notj] += 1
score_fl /= np.maximum(cnt, 1)

def render(score, title, fname):
    fig, ax = plt.subplots(4, 4, figsize=(16, 15))
    for n in range(N):
        a = ax[n // 4, n % 4]
        img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
        sel = np.where(img_of == n)[0]; cid = int(club_of[sel[np.argmax(score[sel])]])
        m = data[n]["label_map"] == cid
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
            (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
        out = (img * 0.35).astype(np.uint8)          # darken all
        out[m] = img[m]                              # keep the pick bright
        b = binary_dilation(find_boundaries(m, mode="outer"), iterations=2)
        out[b] = (255, 0, 255)                        # magenta outline
        a.imshow(out); a.set_title(f"{n:02d}  club {cid}", fontsize=10); a.axis("off")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=105, bbox_inches="tight"); plt.close()

render(score_pd, "PER-DIMENSION (256 matrices → pool → 256-d → scalar)", "perdim_clear.png")
render(score_fl, "STANDARD flat consensus (joint dot-product) — baseline", "flat_clear.png")
print("done")
