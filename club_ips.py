"""
Club-IPS — Inter-image Prototype Selector for clubbed embeddings
================================================================
Adapts the IPS idea from TF-SSD (He et al., CVPR 2026) to our clubbed
super-embeddings. Given a GROUP of images, each already segmented into object
"clubs" with a 256-d embedding per club, Club-IPS finds the clubs that recur
ACROSS the group (the co-salient / common object), and scores every club.

IPS principle: a club is co-salient if it has a strong best-match in EVERY other
image of the group.

  co_score(club c in image n) = mean over other images m of
                                  max over clubs c' in image m of  cos(e_c, e_c')

All club embeddings are L2-normalised, so cosine similarity is a dot product.

Use:
  from club_ips import club_ips
  result = club_ips(group)            # group: list of per-image dicts (see below)
Each per-image dict needs at least:
  "club_super_embeddings": {club_id: (256,) np.float32}   (or "club_embeddings")
Optional for visualisation: "label_map" (H,W), "meta": {"file": ...}
"""

import numpy as np


def _stack(group, key):
    """Stack all club embeddings across the group into P (M,256) + index arrays."""
    vecs, img_of, club_of = [], [], []
    for n, g in enumerate(group):
        emb = g.get(key) or g.get("club_embeddings")
        for cid, v in emb.items():
            v = np.asarray(v, dtype=np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
            img_of.append(n); club_of.append(int(cid))
    return np.stack(vecs), np.array(img_of), np.array(club_of)


def club_ips(group, key="club_super_embeddings"):
    """
    Returns a dict with:
      co_score    (M,)            co-salient score per club (higher = more common)
      img_of      (M,)            image index of each club
      club_of     (M,)            club id of each club
      best_per_image  {n: (club_id, score)}   the top co-salient club in image n
      C           (M,M)           cosine similarity matrix (for inspection)
    """
    P, img_of, club_of = _stack(group, key)
    N = len(group)
    C = P @ P.T                                   # (M, M) cosine similarities

    co = np.zeros(len(P), dtype=np.float32)
    for m in range(len(P)):
        per_img_max = []
        for other in range(N):
            if other == img_of[m]:
                continue                          # ignore same image (intra-image)
            sel = (img_of == other)
            if sel.any():
                per_img_max.append(C[m, sel].max())   # best match in that image
        co[m] = float(np.mean(per_img_max)) if per_img_max else 0.0

    best = {}
    for n in range(N):
        sel = np.where(img_of == n)[0]
        if len(sel):
            b = sel[np.argmax(co[sel])]
            best[n] = (int(club_of[b]), float(co[b]))
    return {"co_score": co, "img_of": img_of, "club_of": club_of,
            "best_per_image": best, "C": C}


def co_salient_clubs(group, key="club_super_embeddings", thresh=None):
    """
    Convenience: return, per image, the set of clubs whose co-salient score is
    above `thresh` (default = the mean co-score across the group). Captures cases
    where the common object spans several clubs, not just the single best one.
    """
    r = club_ips(group, key)
    co, img_of, club_of = r["co_score"], r["img_of"], r["club_of"]
    t = float(co.mean()) if thresh is None else thresh
    out = {}
    for n in range(len(group)):
        sel = np.where(img_of == n)[0]
        out[n] = sorted(int(club_of[i]) for i in sel if co[i] >= t)
    return out, r
