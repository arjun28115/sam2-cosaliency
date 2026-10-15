"""
Multi-instance extension of the final method  —  SAM2-only, general.
====================================================================
The clique picks ONE region per image. A second instance (two clocks in one
image) is recovered by an ABSOLUTE bar calibrated on the winners themselves:

  accept candidate c in image i iff
    - it does not overlap image i's winner
    - its absolute soft-AND group agreement  >=  P25 of the 16 winners' own
      agreements  (scale-free: the group defines its own acceptance level)

Relative-to-own-winner thresholds fail (a weak winner inflates its distractor's
ratio); the absolute bar separates: second instances score like winners,
distractors score like the weakest winner or below.
"""
import numpy as np
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
import v_reencode as vr
from harness import load
from v_final import run
from v_anneal import _qsim
from v_combo import _prep, _cache


def multi_instance(data, picks, quantile=25, top_scan=4, max_extra=2):
    U, cr, ci, cm, sp_img, fg = _prep(data, 3)
    G = np.maximum(U @ U.T, 0.0)
    rows_of_club = {}
    for k, mem in enumerate(cm):
        if len(mem) == 1:
            rows_of_club[(ci[k], next(iter(mem)))] = cr[k]
    N = len(data)
    win_rows = [np.concatenate([rows_of_club[(n, c)] for c in mem])
                for n, (mem, _) in enumerate(picks)]

    def desc(n, members, masked):
        img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
        lm = data[n]["label_map"]
        lmf = np.array(Image.fromarray(lm.astype(np.int32), mode="I")
                       .resize((img.shape[1], img.shape[0]), Image.NEAREST))
        return vr._encode_crop(img, np.isin(lmf, list(members)), masked=masked)

    win_d = {(n, m): desc(n, picks[n][0], m) for n in range(N) for m in [False, True]}

    def softand(n, dr, dm, rows, topk=3):
        """top-k mean of per-winner soft-AND agreement: a true instance has
        near-duplicate matches in the group (peaked profile); a distractor
        matches uniformly mediocrely (flat profile)."""
        vals = []
        for j in range(N):
            if j == n:
                continue
            q = _qsim(G, rows, win_rows[j], 0.3)
            r = max(0.0, float(dr @ win_d[(j, False)]))
            mm = max(0.0, float(dm @ win_d[(j, True)]))
            vals.append(q * r * mm)
        return float(np.mean(sorted(vals, reverse=True)[:topk]))

    wabs = [softand(n, win_d[(n, False)], win_d[(n, True)], win_rows[n]) for n in range(N)]
    bar = float(np.percentile(wabs, quantile))
    print(f"acceptance bar = P{quantile}(winners' top-3) = {bar:.4f}", flush=True)

    extras = [[] for _ in range(N)]
    for n in range(N):
        lm = data[n]["label_map"]
        taken = picks[n][1].copy()
        cands = []
        for k in np.where(ci == n)[0]:
            mask = np.isin(lm, list(cm[k]))
            if (mask & taken).sum() / max(1, mask.sum()) > 0.1:
                continue
            ag = np.mean([_qsim(G, cr[k], win_rows[j], 0.3) for j in range(N) if j != n])
            cands.append((ag, k))
        cands.sort(reverse=True)
        for _, k in cands[:top_scan]:
            if len(extras[n]) >= max_extra:
                break
            mem = sorted(cm[k])
            mask = np.isin(lm, mem)
            if (mask & taken).sum() / max(1, mask.sum()) > 0.1:
                continue
            a = softand(n, desc(n, mem, False), desc(n, mem, True), cr[k])
            if a >= bar:
                extras[n].append((mem, mask, round(a, 4)))
                taken |= mask
                print(f"  img{n:02d}: ACCEPT second instance {mem} (agree {a:.4f})", flush=True)
    return extras


if __name__ == "__main__":
    vr.CACHE_DIR = Path("/Users/arjunsingla/sam2/data/reencode_cache")
    data = load()
    picks = run(data, K=10, q=0.3, lam=0.15)
    extras = multi_instance(data, picks)

    # render combined single+multi instance result
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    from scipy.ndimage import binary_dilation
    fig, ax = plt.subplots(4, 4, figsize=(16, 15))
    for n in range(16):
        a = ax[n // 4, n % 4]
        img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
        mem, m = picks[n]
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
            (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
        out = (img * 0.32).astype(np.uint8); out[m] = img[m]
        b = binary_dilation(find_boundaries(m, mode="outer"), iterations=3)
        out[b] = (0, 255, 60)
        for (mem2, m2, sc) in extras[n]:
            m2 = np.array(Image.fromarray(m2.astype(np.uint8) * 255).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
            out[m2] = img[m2]
            b2 = binary_dilation(find_boundaries(m2, mode="outer"), iterations=3)
            out[b2] = (255, 200, 0)
        extra_txt = f" +{len(extras[n])}" if extras[n] else ""
        a.imshow(out); a.set_title(f"{n:02d}{extra_txt}", fontsize=10); a.axis("off")
    fig.suptitle("Multi-instance co-saliency — green = winner, orange = accepted 2nd instance", fontsize=13)
    plt.tight_layout()
    plt.savefig("/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency/MULTI_instance.png",
                dpi=105, bbox_inches="tight")
    print("saved MULTI_instance.png", flush=True)
