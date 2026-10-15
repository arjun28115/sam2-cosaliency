"""
SAM2-only co-saliency module (vectorized + background penalty + mask output)
============================================================================
Implements the Inter-Image Consensus Score (steps A-D) plus the refinements:
  * Tip #1 — intra-image background penalty (subtract similarity to the image's
             boundary/background clubs) to defeat background dominance.
  * Tip #2 — threshold Score(p) (Otsu or top-K percentile) and map clubs back to
             the pixel grid -> crisp binary co-salient mask per image.
  * Tip #3 — vectorized similarity: one big matmul over all clubs, per-image
             max-pool looped only over N images (not per-superpixel).

Works on any per-club descriptor stored in the pipeline output (default the SAM2
'club_super_embeddings'); pass embed_key to use another (e.g. 'club_dino').

API:
    masks, scores = co_salient_masks(images, embed_key="club_super_embeddings")
where `images` is a list of per-image dicts each with that embedding dict + 'label_map'.
"""

import numpy as np
import torch

try:
    from skimage.filters import threshold_otsu
    _HAS_OTSU = True
except Exception:
    _HAS_OTSU = False


def _border_occupancy(label_map):
    """Fraction of the image border occupied by each club id -> {cid: occ}."""
    H, W = label_map.shape
    bl = 2 * (H + W)
    fr = np.zeros_like(label_map, bool)
    fr[0, :] = fr[-1, :] = fr[:, 0] = fr[:, -1] = True
    occ = {}
    for cid in np.unique(label_map):
        occ[int(cid)] = float(((label_map == cid) & fr).sum()) / bl
    return occ


def co_salient_scores(images, embed_key="club_super_embeddings",
                      bg_lambda=0.8, bg_border=0.12, device="cpu"):
    """
    Returns (score_per_club, img_of, club_of) where score_per_club[k] is the
    background-penalised inter-image consensus Score for club k.
    """
    # ---- flatten all clubs into one matrix (Tip #3: one big matmul) ----
    P, img_of, club_of, border = [], [], [], []
    for n, g in enumerate(images):
        occ = _border_occupancy(g["label_map"])
        for cid, v in g[embed_key].items():
            v = np.asarray(v, np.float32)
            P.append(v / (np.linalg.norm(v) + 1e-8))     # Step A: L2-normalise
            img_of.append(n); club_of.append(int(cid)); border.append(occ[int(cid)])
    P = torch.tensor(np.stack(P), device=device)          # (M, D)
    img_of = np.array(img_of); club_of = np.array(club_of); border = np.array(border)
    N = len(images); M = len(P)

    C = (P @ P.T).cpu().numpy()                            # Step B: full similarity (one matmul)

    # ---- inter-image consensus, vectorized over the N images ----
    cons = np.zeros(M); cnt = np.zeros(M)
    for j in range(N):
        colmask = (img_of == j)
        if not colmask.any():
            continue
        colmax = C[:, colmask].max(axis=1)                # Step C: max over q in I_j
        notj = (img_of != j)
        cons[notj] += colmax[notj]; cnt[notj] += 1
    score = cons / np.maximum(cnt, 1)                     # Step D: average over j != i

    # ---- Tip #1: intra-image background penalty ----
    pen = np.zeros(M)
    for i in range(N):
        ini = (img_of == i); bnd = ini & (border > bg_border)
        if bnd.any():
            pen[ini] = C[np.ix_(ini, bnd)].max(axis=1)    # similarity to own background clubs
    score = score - bg_lambda * pen
    return score, img_of, club_of


def co_salient_masks(images, embed_key="club_super_embeddings",
                     bg_lambda=0.8, mode="otsu", topk=0.15, device="cpu"):
    """
    Returns (masks, per_image_scores):
      masks[n]              : (H, W) bool co-salient mask for image n
      per_image_scores[n]   : {club_id: score}
    `mode`: "otsu" (auto threshold) or "topk" (keep clubs in top `topk` fraction).
    """
    score, img_of, club_of = co_salient_scores(images, embed_key, bg_lambda, device=device)
    masks, per_image = [], []
    for n, g in enumerate(images):
        sel = np.where(img_of == n)[0]
        sc = score[sel]; cids = club_of[sel]
        per_image.append({int(c): float(s) for c, s in zip(cids, sc)})
        if len(sc) == 0:
            masks.append(np.zeros_like(g["label_map"], bool)); continue
        if mode == "otsu" and _HAS_OTSU and len(np.unique(sc)) > 1:
            try:
                t = threshold_otsu(sc)
            except Exception:
                t = np.quantile(sc, 1 - topk)
        else:
            t = np.quantile(sc, 1 - topk)
        keep = cids[sc >= t]
        # Tip #2: map selected clubs back to the pixel grid -> crisp mask
        masks.append(np.isin(g["label_map"], keep))
    return masks, per_image


# ── Dual-stream fusion (semantic SigLIP + geometric SAM2) ─────────────────────

def _flatten_dual(group, sem_key, geo_key):
    """Flatten two per-club descriptors over the group, keeping clubs present in both."""
    Ps, Pg, img_of, club_of, border = [], [], [], [], []
    for n, g in enumerate(group):
        lm = g["label_map"]; H, W = lm.shape; bl = 2 * (H + W)
        fr = np.zeros_like(lm, bool); fr[0, :] = fr[-1, :] = fr[:, 0] = fr[:, -1] = True
        occ = {int(c): float(((lm == c) & fr).sum()) / bl for c in np.unique(lm)}
        sem, geo = g[sem_key], g[geo_key]
        for cid in sem:
            if cid not in geo:
                continue
            s = np.asarray(sem[cid], np.float32); gg = np.asarray(geo[cid], np.float32)
            Ps.append(s / (np.linalg.norm(s) + 1e-8))
            Pg.append(gg / (np.linalg.norm(gg) + 1e-8))
            img_of.append(n); club_of.append(int(cid)); border.append(occ[int(cid)])
    return (np.stack(Ps), np.stack(Pg), np.array(img_of), np.array(club_of), np.array(border))


def _consensus_with_penalty(C, img_of, border, bg_lambda, bg_border=0.12):
    """Inter-image consensus (A-D) minus the intra-image background penalty, given a similarity matrix C."""
    M = len(C); N = int(img_of.max()) + 1
    cons = np.zeros(M); cnt = np.zeros(M)
    for j in range(N):
        col = (img_of == j)
        if not col.any():
            continue
        cmax = C[:, col].max(axis=1); notj = (img_of != j)
        cons[notj] += cmax[notj]; cnt[notj] += 1
    score = cons / np.maximum(cnt, 1)
    pen = np.zeros(M)
    for i in range(N):
        ini = (img_of == i); bnd = ini & (border > bg_border)
        if bnd.any():
            pen[ini] = C[np.ix_(ini, bnd)].max(axis=1)
    return score - bg_lambda * pen


def co_salient_scores_dual(group, fusion="score", alpha=0.7,
                           sem_key="club_semantic_embeddings",
                           geo_key="club_embeddings", bg_lambda=0.8, sam_voice=np.sqrt(3.0)):
    """
    Dual-stream co-saliency.
      fusion="score"  (Option A): S = alpha*S_sem + (1-alpha)*S_geo, then consensus+penalty.
      fusion="concat" (Option B): club vector = [sem || sam_voice*geo], L2-norm, then consensus+penalty.
    Returns (score, img_of, club_of).
    """
    Ps, Pg, img_of, club_of, border = _flatten_dual(group, sem_key, geo_key)
    if fusion == "score":
        C = alpha * (Ps @ Ps.T) + (1.0 - alpha) * (Pg @ Pg.T)
    elif fusion == "concat":
        Pf = np.concatenate([Ps, sam_voice * Pg], axis=1)
        Pf = Pf / (np.linalg.norm(Pf, axis=1, keepdims=True) + 1e-8)
        C = Pf @ Pf.T
    else:
        raise ValueError("fusion must be 'score' or 'concat'")
    score = _consensus_with_penalty(C, img_of, border, bg_lambda)
    return score, img_of, club_of
