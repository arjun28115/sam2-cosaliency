"""
Spectral co-saliency with a foreground (RBD) prior  —  SAM2-only, general.
==========================================================================
No category knowledge, no per-object constants. Two general principles:

  1. FOREGROUND-NESS (RBD boundary connectivity):
        BndCon(c) = (# club pixels on the image frame) / sqrt(area)
        fg(c)     = exp( -BndCon^2 / (2 sigma^2) )
     Backgrounds hug the image border -> high BndCon -> fg~0.
     Compact interior objects -> BndCon~0 -> fg~1.

  2. MUTUAL CONSENSUS (dominant set / principal eigenvector):
        A club is co-salient iff it is similar to OTHER clubs that are
        themselves co-salient. That fixed point is the top eigenvector of
        the foreground-weighted cross-image affinity:
            C   = relu(P P^T),  zeroed on same-image blocks
            A   = diag(fg) C diag(fg)
            x   = principal eigenvector of A     (power iteration)
     One-directional max-pooling rewards any generic region that matches
     *something* everywhere; the eigenvector rewards only the single
     recurring, mutually-consistent cluster -> the co-salient object.
"""
import numpy as np


def _flatten(data, key="club_super_embeddings"):
    P, img_of, club_of = [], [], []
    for n, g in enumerate(data):
        for cid, v in g[key].items():
            v = np.asarray(v, np.float32)
            P.append(v / (np.linalg.norm(v) + 1e-8))
            img_of.append(n); club_of.append(int(cid))
    return np.stack(P), np.array(img_of), np.array(club_of)


def foreground_prior(data):
    """RBD boundary-connectivity foreground weight per club, flattened to match _flatten order."""
    fg = []
    bcs = []
    for g in data:
        lm = g["label_map"]; H, W = lm.shape
        frame = np.zeros_like(lm, bool)
        frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
        for cid in g["club_super_embeddings"].keys():
            m = lm == int(cid)
            area = float(m.sum()) + 1e-6
            bnd = float((m & frame).sum())
            bcs.append(bnd / np.sqrt(area))
    bcs = np.array(bcs)
    pos = bcs[bcs > 0]
    sigma = np.median(pos) if len(pos) else 1.0
    fg = np.exp(-(bcs ** 2) / (2 * sigma ** 2 + 1e-8))
    return fg.astype(np.float32)


def spectral_cosaliency(data, key="club_super_embeddings", iters=100):
    P, img_of, club_of = _flatten(data, key)
    N = int(img_of.max()) + 1
    fg = foreground_prior(data)

    C = P @ P.T
    C = np.maximum(C, 0.0)                       # keep positive agreement only
    same = img_of[:, None] == img_of[None, :]
    C[same] = 0.0                                # cross-image consensus only
    A = (fg[:, None] * C) * fg[None, :]          # foreground-weighted affinity

    # power iteration -> principal eigenvector (dominant mutually-consistent cluster)
    x = fg.copy() + 1e-6
    for _ in range(iters):
        x = A @ x
        n = np.linalg.norm(x)
        if n < 1e-12:
            break
        x = x / n
    x = np.abs(x)
    score = x * fg                               # anchor the eigenvector on foreground
    return score, img_of, club_of


def select(score, img_of, club_of, data):
    N = int(img_of.max()) + 1
    out = []
    for n in range(N):
        sel = np.where(img_of == n)[0]
        cid = int(club_of[sel[np.argmax(score[sel])]])
        out.append((cid, data[n]["label_map"] == cid))
    return out
