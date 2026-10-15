import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from cosal_perdim import per_dimension_consensus, select_per_image, _flatten

DATA = "/tmp/clock_grey.npy"
OUT  = "/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency"
data = np.load(DATA, allow_pickle=True)
N = len(data)

# ---------- per-dimension method ----------
Z, score_pd, img_of, club_of, Sim = per_dimension_consensus(data)
print("Z (per-club 256-d signature):", Z.shape, " scalar score:", score_pd.shape)

# ---------- standard flat consensus (baseline) ----------
P, img2, club2 = _flatten(data)
C = P @ P.T
score_flat = np.zeros(len(P)); cnt = np.zeros(len(P))
for j in range(N):
    col = img2 == j
    cmax = C[:, col].max(1); notj = img2 != j
    score_flat[notj] += cmax[notj]; cnt[notj] += 1
score_flat /= np.maximum(cnt, 1)

def pick(score):
    out = []
    for n in range(N):
        sel = np.where(img_of == n)[0]
        out.append(int(club_of[sel[np.argmax(score[sel])]]))
    return out
pd_pick, fl_pick = pick(score_pd), pick(score_flat)
print("per-dim picks :", pd_pick)
print("flat    picks :", fl_pick)
print("agree on", sum(a == b for a, b in zip(pd_pick, fl_pick)), "/", N)

# ---------- viz 1: a few per-dimension N x N image-image matrices + pooled ----------
var = Sim.reshape(Sim.shape[0], -1).var(1)
top = np.argsort(-var)[:5]                       # 5 most informative channels
pooled = Sim.mean(0)                             # pool all 256 -> N x N
fig, ax = plt.subplots(1, 6, figsize=(22, 4))
for k, d in enumerate(top):
    ax[k].imshow(Sim[d], cmap="magma"); ax[k].set_title(f"dim {d}\n(N×N)", fontsize=10)
    ax[k].set_xticks([]); ax[k].set_yticks([])
ax[5].imshow(pooled, cmap="magma"); ax[5].set_title("pooled\n(mean of 256)", fontsize=11)
ax[5].set_xticks([]); ax[5].set_yticks([])
fig.suptitle("Per-dimension image×image similarity — 256 matrices, then pooled", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUT}/perdim_matrices.png", dpi=110, bbox_inches="tight"); plt.close()

# ---------- viz 2: selected co-salient club per image (per-dim method) ----------
picks = select_per_image(score_pd, img_of, club_of, data)
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for n in range(N):
    a = ax[n // 4, n % 4]
    img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
    cid, m = picks[n]
    from PIL import Image as I
    m = np.array(I.fromarray(m.astype(np.uint8) * 255).resize((img.shape[1], img.shape[0]), I.NEAREST)) > 127
    ov = img.copy(); ov[m] = (0.45 * ov[m] + 0.55 * np.array([0, 255, 120])).astype(np.uint8)
    a.imshow(ov); a.set_title(f"{n:02d}  club {cid}", fontsize=10)
    a.axis("off")
fig.suptitle("Per-dimension co-saliency — selected club per image", fontsize=15)
plt.tight_layout(); plt.savefig(f"{OUT}/perdim_selection.png", dpi=105, bbox_inches="tight"); plt.close()

# ---------- viz 3: the pooled 256-d signature of the winning club, few images ----------
fig, ax = plt.subplots(4, 1, figsize=(15, 9))
for r, n in enumerate([0, 3, 6, 15]):
    sel = np.where(img_of == n)[0]
    best = sel[np.argmax(score_pd[sel])]
    ax[r].bar(np.arange(256), Z[best], color="teal", width=1.0)
    ax[r].set_title(f"image {n:02d}, club {club_of[best]} — pooled 256-d co-saliency signature", fontsize=10)
    ax[r].set_xlim(0, 255)
plt.tight_layout(); plt.savefig(f"{OUT}/perdim_signature.png", dpi=105, bbox_inches="tight"); plt.close()

print("saved: perdim_matrices.png, perdim_selection.png, perdim_signature.png")
