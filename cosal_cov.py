"""
Coverage co-saliency  —  SAM2-only, general.
============================================
Two general principles, no category constants:

  fg(c)  : RBD boundary-connectivity foreground weight (kills background).
  cov(p) : robust CROSS-IMAGE COVERAGE. For club p in image i,
              m_j(p) = max_{q in j} relu(cos(p,q))      # best match in image j
           A co-salient object must have a good match in EVERY other image, so
           aggregate by the WORST images, not the mean:
              cov(p) = mean of the lowest  ceil(low_frac*(N-1))  values of {m_j}.
           A distractor that recurs in only a few images (a cup in 5/16) scores
           great on those but ~0 on the rest -> low coverage. The ubiquitous
           object (present in all) keeps a floor everywhere -> high coverage.

  score(p) = fg(p) * cov(p) ** gamma
"""
import numpy as np
from cosal_spectral import _flatten, foreground_prior


def coverage_cosaliency(data, key="club_super_embeddings",
                        low_frac=0.5, gamma=1.0, fg_pow=1.0):
    P, img_of, club_of = _flatten(data, key)
    N = int(img_of.max()) + 1
    fg = foreground_prior(data) ** fg_pow

    S = np.maximum(P @ P.T, 0.0)                    # positive cosine agreement
    M = len(P)

    # best match of each club in each other image  -> m[p, j]
    m = np.zeros((M, N), np.float32)
    for j in range(N):
        col = img_of == j
        m[:, j] = S[:, col].max(1) if col.any() else 0.0

    cov = np.zeros(M, np.float32)
    for p in range(M):
        others = np.delete(m[p], img_of[p])         # drop own image
        k = max(1, int(np.ceil(low_frac * len(others))))
        cov[p] = np.sort(others)[:k].mean()         # mean of the WORST k images
    score = fg * (cov ** gamma)
    return score, img_of, club_of


def select(score, img_of, club_of, data):
    N = int(img_of.max()) + 1
    out = []
    for n in range(N):
        sel = np.where(img_of == n)[0]
        cid = int(club_of[sel[np.argmax(score[sel])]])
        out.append((cid, data[n]["label_map"] == cid))
    return out
