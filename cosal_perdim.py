"""
Per-dimension co-saliency
==========================
Idea (user's formulation): instead of collapsing all 256 channels into ONE
similarity number, keep every embedding dimension separate.

  For each dimension d in {0..255}:
     cell(i,j) = E_i[:,d]  outer  E_j[:,d]          # (M_i x M_j) sub-matrix
     per-club consensus:
        z_d(p in i) = mean_{j != i} max_{q in j} ( E_i[p,d] * E_j[q,d] )
  Stack over d  -> Z(p) in R^256   (the "pool back to 256-d" signature)
  scalar score  = sum_d Z(p,d)      -> pick the co-salient club per image

Closed form for the per-dimension max (no M x M matrices needed):
  max_{q in j} E_i[p,d]*E_j[q,d] =
        E_i[p,d] * ( imgmax[j,d] if E_i[p,d] >= 0 else imgmin[j,d] )
where imgmax[j,d] / imgmin[j,d] are the max / min of channel d over image j's clubs.
"""
import numpy as np

EMB_KEY = "club_super_embeddings"


def _flatten(data, key=EMB_KEY):
    P, img_of, club_of = [], [], []
    for n, g in enumerate(data):
        for cid, v in g[key].items():
            v = np.asarray(v, np.float32)
            P.append(v / (np.linalg.norm(v) + 1e-8))      # L2-normalise (cosine geometry)
            img_of.append(n); club_of.append(int(cid))
    return np.stack(P), np.array(img_of), np.array(club_of)


def per_dimension_consensus(data, key=EMB_KEY):
    """
    Returns
      Z         : (M, 256) per-club per-dimension consensus signature
      score     : (M,)     pooled scalar score  = Z.sum(1)
      img_of    : (M,)     image index of each club
      club_of   : (M,)     club id of each club
      Sim       : (256, N, N)  per-dimension image-image similarity matrices
                              Sim[d,i,j] = max_{p in i, q in j} E_i[p,d]*E_j[q,d]
    """
    P, img_of, club_of = _flatten(data, key)
    M, Dd = P.shape
    N = int(img_of.max()) + 1

    # per-image per-channel extrema
    imgmax = np.full((N, Dd), -np.inf, np.float32)
    imgmin = np.full((N, Dd),  np.inf, np.float32)
    for j in range(N):
        rows = P[img_of == j]
        imgmax[j] = rows.max(0)
        imgmin[j] = rows.min(0)

    pos = P >= 0                                          # (M,256) sign of the anchor channel

    # A[p, j, d] = max_{q in j} E[p,d]*E[q,d]   (closed form)
    #   = P[p,d] * (imgmax[j,d] if P[p,d]>=0 else imgmin[j,d])
    chosen = np.where(pos[:, None, :], imgmax[None, :, :], imgmin[None, :, :])  # (M,N,256)
    A = P[:, None, :] * chosen                            # (M,N,256)

    # exclude self-image j == img_of[p], average over the remaining N-1 images
    Z = np.empty((M, Dd), np.float32)
    for p in range(M):
        i = img_of[p]
        keep = np.ones(N, bool); keep[i] = False
        Z[p] = A[p, keep, :].mean(0)                      # (256,)  <- pooled back to 256-d
    score = Z.sum(1)                                      # scalar co-saliency

    # per-dimension N x N image-image similarity (max club pair per channel)
    Sim = np.empty((Dd, N, N), np.float32)
    for i in range(N):
        Pi = P[img_of == i]                               # (Mi,256)
        for j in range(N):
            Pj = P[img_of == j]                           # (Mj,256)
            # max over p,q of Pi[p,d]*Pj[q,d] per channel d
            # = max(imax_i * imax_j, imin_i * imin_j)   (largest same-sign product)
            imax_i, imin_i = Pi.max(0), Pi.min(0)
            imax_j, imin_j = Pj.max(0), Pj.min(0)
            Sim[:, i, j] = np.maximum(imax_i * imax_j, imin_i * imin_j)
    return Z, score, img_of, club_of, Sim


def select_per_image(score, img_of, club_of, data):
    """argmax-scoring club per image -> list of (club_id, pixel_mask)."""
    N = int(img_of.max()) + 1
    out = []
    for n in range(N):
        sel = np.where(img_of == n)[0]
        best = sel[np.argmax(score[sel])]
        cid = int(club_of[best])
        out.append((cid, data[n]["label_map"] == cid))
    return out
