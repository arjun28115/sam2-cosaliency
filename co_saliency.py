"""
Inter-Image Consensus Co-Saliency Score
========================================
Faithful implementation of the scoring strategy:

  Given image set I = {I_1, ..., I_N}, each image i with superpixel/club
  embeddings E^(i) = {e_1^(i), ..., e_{M_i}^(i)}, e in R^D.

  A) normalise:        ê = e / ||e||_2
  B) similarity:       S(p,q) = ê_p^(i) · ê_q^(j)
  C) image consensus:  C(p, I_j) = max_{q in I_j} ( ê_p^(i) · ê_q^(j) )
  D) co-saliency:      Score(p) = (1/(N-1)) * sum_{j != i} C(p, I_j)

A superpixel is co-salient if it has a strong best-match in every other image.
The scorer is descriptor-agnostic: pass SAM2 clubbed embeddings, DINO prototypes,
or anything in R^D.
"""

import numpy as np


def co_saliency_scores(embeddings_per_image, reduce="avg"):
    """
    Parameters
    ----------
    embeddings_per_image : list of N arrays, the i-th of shape (M_i, D)
        Per-image superpixel/club embeddings (any descriptor).
    reduce : "avg" or "sum"
        Step D aggregation over the N-1 other images.

    Returns
    -------
    scores : list of N arrays, the i-th of shape (M_i,)
        Score(p) for every superpixel p, matching the input layout.
    """
    N = len(embeddings_per_image)

    # Step A — L2-normalise every embedding (dot product == cosine similarity)
    E = []
    for Ei in embeddings_per_image:
        Ei = np.asarray(Ei, dtype=np.float32)
        Ei = Ei / (np.linalg.norm(Ei, axis=1, keepdims=True) + 1e-8)
        E.append(Ei)

    scores = []
    for i in range(N):
        Ei = E[i]                                   # (M_i, D)
        per_image_consensus = []                    # list of (M_i,) arrays, one per other image
        for j in range(N):
            if j == i:
                continue
            # Step B — full cross-image similarity matrix  S in R^{M_i x M_j}
            S = Ei @ E[j].T                          # ê_p^(i) · ê_q^(j) for all p,q
            # Step C — max-pool over q in I_j  ->  C(p, I_j) in R^{M_i}
            per_image_consensus.append(S.max(axis=1))
        if per_image_consensus:
            Cmat = np.stack(per_image_consensus, axis=1)   # (M_i, N-1)
            # Step D — aggregate over the N-1 other images
            scores.append(Cmat.mean(1) if reduce == "avg" else Cmat.sum(1))
        else:
            scores.append(np.zeros(len(Ei), dtype=np.float32))
    return scores


def select_co_salient(embeddings_per_image, reduce="avg"):
    """Convenience: return argmax-scoring superpixel index per image + the scores."""
    scores = co_saliency_scores(embeddings_per_image, reduce)
    best = [int(np.argmax(s)) if len(s) else -1 for s in scores]
    return best, scores
