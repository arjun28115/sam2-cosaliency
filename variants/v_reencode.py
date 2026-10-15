"""
SAM2 crop re-encoding stage (SAM2-only, general).
=================================================
The superpixel-level descriptors hit their ceiling: a green mug's texture is
indistinguishable from dark clock bodies at patch scale. Second opinion from
the SAME model at OBJECT scale:

  1. shortlist finalist candidates per image (top-K by evidence  U  top-K by
     agreement with the evidence-gated picks) — small, high-recall set.
  2. crop each finalist from the ORIGINAL full-res image (bbox + margin),
     re-encode the crop with SAM2's image encoder, mean-pool image_embed over
     the candidate mask -> one object-centric 256-d descriptor per finalist.
     Small objects are upsampled to canonical scale — their smallness penalty
     disappears; whole-object shape (twin bells, round face) dominates.
  3. global consensus clique on the re-encoded descriptors:
     assignment a maximizing  sum_{i<j} cos(d_ai, d_aj) + lam * evidence.

No category knowledge anywhere; SAM2 is the only model used.
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
from harness import candidates, rbd_fg_candidates
from v_base import evidence, gated_assignment, _unit
from v_combo import _prep, _cache, weighted_set_sim

CACHE_DIR = Path("/Users/arjunsingla/sam2/data/reencode_cache")
CACHE_DIR.mkdir(exist_ok=True)

_pred = None


def _predictor():
    global _pred
    if _pred is None:
        import torch
        os.chdir("/Users/arjunsingla/sam2")
        sys.path.insert(0, "/Users/arjunsingla/sam2")   # repo package wins over site-packages
        for m in [m for m in list(sys.modules) if m == "sam2" or m.startswith("sam2.")]:
            del sys.modules[m]
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _pred = SAM2ImagePredictor(build_sam2(
            "configs/sam2.1/sam2.1_hiera_l", "checkpoints/sam2.1_hiera_large.pt", device=device))
        print("SAM2 loaded on", device, flush=True)
    return _pred


def _shortlist(data, K=6, small_k=0):
    """Finalists per image: top-K by evidence U top-K by agreement with the
    evidence-gated picks U (optionally) top-small_k by agreement among SMALL
    candidates (bottom size-tercile) — evidence favours large regions, so tiny
    co-salient objects (dice) never reach the finalist stage without this.
    Returns (per-image list of candidate indices, prep tuple)."""
    U, cr, ci, cm, sp_img, fg = _prep(data, 3)
    key = ("sim", 0.0, 0.1)
    if key not in _cache:
        _cache[key] = weighted_set_sim(U, cr, ci, np.ones(len(U), np.float32))
    sim, maxub, G = _cache[key]
    ev = evidence(sim, ci, fg, 0.35, 1.0)
    a = gated_assignment(sim, ci, ev, 0.5)
    N = int(ci.max()) + 1

    area = None
    if small_k > 0:
        club_area = {}
        for n, g in enumerate(data):
            lm = g["label_map"]
            ids, cnts = np.unique(lm, return_counts=True)
            for c, s in zip(ids, cnts):
                club_area[(n, int(c))] = int(s)
        area = np.array([sum(club_area.get((ci[k], c), 0) for c in cm[k])
                         for k in range(len(cm))], np.float64)

    short = []
    for n in range(N):
        sel = np.where(ci == n)[0]
        others = [a[j] for j in range(N) if j != n]
        agree = sim[np.ix_(sel, others)].mean(1)
        top_ev = sel[np.argsort(-ev[sel])[:K]]
        top_ag = sel[np.argsort(-agree)[:K]]
        keep = set(top_ev) | set(top_ag)
        if small_k > 0:
            # PEER-ranked strata: within each size tercile, rank candidates by
            # their best match to the SAME tercile in other images (a moon
            # agrees with moons, a die with dice) — ranking against the
            # evidence picks is circular (they are landscapes/boards).
            qs = np.percentile(area[sel], [33, 66])
            for lo, hi in [(0, qs[0]), (qs[0], qs[1])]:      # small + mid strata
                sm = sel[(area[sel] > lo) & (area[sel] <= hi)]
                if not len(sm):
                    continue
                peer_cols = []
                for j in range(N):
                    if j == n:
                        continue
                    sj = np.where(ci == j)[0]
                    qj = np.percentile(area[sj], [33, 66])
                    ratio = (lo / (qs[0] + 1e-9), hi / (qs[1] + 1e-9))
                    peer_cols.append(sj[(area[sj] > (qj[0] if lo > 0 else 0)) &
                                        (area[sj] <= (qj[1] if lo > 0 else qj[0]))])
                peer_cols = np.concatenate(peer_cols) if peer_cols else np.array([], int)
                if len(peer_cols):
                    ag_sm = sim[np.ix_(sm, peer_cols)].max(1)
                else:
                    ag_sm = sim[np.ix_(sm, others)].mean(1)
                keep |= set(sm[np.argsort(-ag_sm)[:small_k]])
        short.append(sorted(keep))
    return short, (U, cr, ci, cm, fg, sim, ev, a)


def _encode_crop(img, mask_full, margin=0.15, masked=False):
    """Crop bbox+margin from full-res img, SAM2-encode, mean-pool image_embed
    over the mask region -> 256-d L2-normalised descriptor."""
    import torch
    pred = _predictor()
    ys, xs = np.where(mask_full)
    H, W = mask_full.shape
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    my = int((y1 - y0) * margin); mx = int((x1 - x0) * margin)
    y0, y1 = max(0, y0 - my), min(H, y1 + my)
    x0, x1 = max(0, x0 - mx), min(W, x1 + mx)
    crop = img[y0:y1, x0:x1].copy()
    cmask = mask_full[y0:y1, x0:x1]
    if masked:
        fill = crop.reshape(-1, 3).mean(0).astype(np.uint8)
        crop[~cmask] = fill
    with torch.inference_mode():
        pred.set_image(crop)
        fe = pred._features["image_embed"][0].cpu().numpy()      # (256, 64, 64)
    m64 = np.array(Image.fromarray(cmask.astype(np.uint8) * 255)
                   .resize((64, 64), Image.NEAREST)) > 127
    if m64.sum() < 4:
        m64 = np.ones((64, 64), bool)
    v = fe[:, m64].mean(1)
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def _descriptors(data, short, ci, cm, masked=False):
    """Re-encode every shortlisted candidate (disk-cached)."""
    tagm = "m1" if masked else "m0"
    descs = {}
    for n, kl in enumerate(short):
        img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
        lm = data[n]["label_map"]
        lm_full = np.array(Image.fromarray(lm.astype(np.int32), mode="I")
                           .resize((img.shape[1], img.shape[0]), Image.NEAREST))
        for k in kl:
            mem = sorted(cm[k])
            f = CACHE_DIR / f"img{n}_{'_'.join(map(str, mem))}_{tagm}.npy"
            if f.exists():
                descs[k] = np.load(f)
                continue
            mask_full = np.isin(lm_full, mem)
            if not mask_full.any():
                descs[k] = np.zeros(256, np.float32); continue
            v = _encode_crop(img, mask_full, masked=masked)
            np.save(f, v); descs[k] = v
        print(f"encoded img {n} ({len(kl)} finalists)", flush=True)
    return descs


def run(data, K=6, masked=False, lam=0.25, restarts=6):
    short, (U, cr, ci, cm, fg, sim, ev, a0) = _shortlist(data, K)
    descs = _descriptors(data, short, ci, cm, masked)
    N = len(short)
    evu = _unit(ev)

    def pair(x, y):
        return float(descs[x] @ descs[y])

    def ascend(a):
        for _ in range(20):
            changed = False
            for i in range(N):
                others = [a[j] for j in range(N) if j != i]
                vals = [np.mean([pair(k, o) for o in others]) + lam * evu[k] for k in short[i]]
                b = short[i][int(np.argmax(vals))]
                if b != a[i]:
                    a[i] = b; changed = True
            if not changed:
                break
        return a

    def objective(a):
        s = sum(pair(a[i], a[j]) for i in range(N) for j in range(i + 1, N))
        return s + lam * sum(evu[a[i]] for i in range(N))

    best_a, best_J = None, -1e18
    inits = []
    a_init = [a0[n] if a0[n] in short[n] else short[n][0] for n in range(N)]
    inits.append(list(a_init))
    for r in range(restarts - 1):
        inits.append([short[n][np.argsort([-evu[k] for k in short[n]])[min(r, len(short[n]) - 1)]]
                      for n in range(N)])
    for a in inits:
        a = ascend(list(a)); J = objective(a)
        if J > best_J:
            best_J, best_a = J, list(a)

    picks = []
    for n in range(N):
        mem = sorted(cm[best_a[n]]); lm = data[n]["label_map"]
        picks.append((mem, np.isin(lm, mem)))
    return picks


if __name__ == "__main__":
    from harness import load, evaluate, save_results
    data = load()
    results = []
    for masked in [False, True]:
        for lam in [0.0, 0.25]:
            picks = run(data, K=10, masked=masked, lam=lam)
            hits, misses, det = evaluate(picks, data)
            print(f"[reencode] masked={masked} lam={lam} -> {len(hits)}/16 miss={misses}", flush=True)
            results.append({"params": {"masked": masked, "lam": lam},
                            "hits": len(hits), "misses": misses, "detail": det})
    results.sort(key=lambda r: -r["hits"])
    save_results(results, "/Users/arjunsingla/sam2/variants/results_reencode.json")
