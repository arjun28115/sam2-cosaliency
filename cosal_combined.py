"""
Combined co-saliency  —  SAM2-only, general (no category constants).
====================================================================
Three orthogonal, general signals multiplied:

  fg(c)   RBD boundary-connectivity foreground weight   -> not-background
  eig(p)  principal eigenvector of the fg-weighted cross-image affinity
          -> MUTUAL CONSISTENCY (a club is co-salient iff similar to other
             co-salient clubs); suppresses generic backgrounds.
  cov(p)  robust cross-image COVERAGE = mean of the worst ceil(low*(N-1))
          per-image best-matches -> UBIQUITY; suppresses tight distractor
          clusters that recur in only a few images (a cup in 5/16 images).

  score(p) = norm(eig) * norm(cov) ** gamma

Each term removes a different failure mode; the product keeps only clubs that
are foreground AND mutually-consistent AND present in (almost) every image —
which, for any group sharing one common object, is that object.
"""
import numpy as np
from cosal_spectral import _flatten, foreground_prior


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (v.max() + 1e-12)


def combined_cosaliency(data, key="club_super_embeddings",
                        low_frac=0.5, gamma=1.0, fg_pow=1.0, iters=100):
    P, img_of, club_of = _flatten(data, key)
    N = int(img_of.max()) + 1
    M = len(P)
    fg = foreground_prior(data) ** fg_pow

    S = np.maximum(P @ P.T, 0.0)
    same = img_of[:, None] == img_of[None, :]

    # --- eigenvector (mutual consistency), foreground-weighted, cross-image only ---
    C = S.copy(); C[same] = 0.0
    A = (fg[:, None] * C) * fg[None, :]
    x = fg.copy() + 1e-6
    for _ in range(iters):
        x = A @ x
        nrm = np.linalg.norm(x)
        if nrm < 1e-12:
            break
        x = x / nrm
    eig = np.abs(x) * fg

    # --- coverage (ubiquity): mean of the worst k per-image best-matches ---
    m = np.zeros((M, N), np.float32)
    for j in range(N):
        col = img_of == j
        m[:, j] = S[:, col].max(1) if col.any() else 0.0
    cov = np.zeros(M, np.float32)
    for p in range(M):
        others = np.delete(m[p], img_of[p])
        k = max(1, int(np.ceil(low_frac * len(others))))
        cov[p] = np.sort(others)[:k].mean()
    cov = cov * fg                                   # coverage of foreground only

    score = _unit(eig) * (_unit(cov) ** gamma)
    return score, img_of, club_of


def select(score, img_of, club_of, data):
    N = int(img_of.max()) + 1
    out = []
    for n in range(N):
        sel = np.where(img_of == n)[0]
        cid = int(club_of[sel[np.argmax(score[sel])]])
        out.append((cid, data[n]["label_map"] == cid))
    return out
