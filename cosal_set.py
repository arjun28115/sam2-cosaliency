"""
Set-based co-saliency  —  SAM2-only, general.
=============================================
Each club is a SET of superpixel embeddings (not one mean vector), so internal
structure is preserved. Club-club similarity is a symmetric (min) Chamfer:

   dir(a->b) = mean_{u in a} max_{v in b} relu(cos(u,v))
   sim(a,b)  = min( dir(a->b), dir(b->a) )

The min is the discriminator: a uniform distractor (all-white cup) matches a
structured object's sub-part (white clock face) in ONE direction, but the
object's other parts (dark rim, bells, hands) have no match back -> min is low.
So a distractor only scores high against its OWN kind, collapsing its coverage
in images where it is absent.

On top of sim, the same three general signals as before:
   fg (RBD foreground) x eig (mutual consistency) x cov (ubiquity).
"""
import numpy as np
from cosal_spectral import foreground_prior


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (v.max() + 1e-12)


def _build_clubs(data, key="club_super_embeddings"):
    """Global club table with superpixel row ranges into a stacked embedding matrix U."""
    U, sp_img = [], []
    clubs = []            # (img, cid, row_indices)
    club_img, club_id = [], []
    t = 0
    for n, g in enumerate(data):
        se = np.asarray(g["super_embeddings"], np.float32)
        se = se / (np.linalg.norm(se, axis=1, keepdims=True) + 1e-8)
        sp_ids = list(np.asarray(g["superpixel_ids"]))
        sp2c = g["sp_to_club"]
        # rows (into global U) for each club in this image
        by_club = {}
        for k, spid in enumerate(sp_ids):
            cid = int(sp2c[int(spid)])
            by_club.setdefault(cid, []).append(t + k)
        U.append(se); sp_img += [n] * len(sp_ids); t += len(sp_ids)
        for cid in g[key].keys():
            cid = int(cid)
            if cid in by_club:
                clubs.append(np.array(by_club[cid])); club_img.append(n); club_id.append(cid)
    U = np.concatenate(U, 0)
    return U, np.array(sp_img), clubs, np.array(club_img), np.array(club_id)


def set_similarity(data, key="club_super_embeddings"):
    U, sp_img, clubs, club_img, club_id = _build_clubs(data, key)
    G = np.maximum(U @ U.T, 0.0)                      # superpixel cosine, positive part
    C = len(clubs)

    # maxub[u, b] = max over superpixels v in club b of G[u,v]
    T = len(U)
    maxub = np.empty((T, C), np.float32)
    for b in range(C):
        maxub[:, b] = G[:, clubs[b]].max(1)

    # dir[a,b] = mean over u in club a of maxub[u,b]
    dirm = np.empty((C, C), np.float32)
    for a in range(C):
        dirm[a] = maxub[clubs[a]].mean(0)
    sim = np.minimum(dirm, dirm.T)                    # symmetric min-Chamfer

    same = club_img[:, None] == club_img[None, :]
    sim[same] = 0.0
    return sim, club_img, club_id


def set_cosaliency(data, key="club_super_embeddings",
                   low_frac=0.5, gamma=1.0, fg_pow=1.0, iters=100):
    sim, img_of, club_of = set_similarity(data, key)
    N = int(img_of.max()) + 1
    M = len(sim)
    fg = foreground_prior(data) ** fg_pow

    A = (fg[:, None] * sim) * fg[None, :]
    x = fg.copy() + 1e-6
    for _ in range(iters):
        x = A @ x
        nrm = np.linalg.norm(x)
        if nrm < 1e-12:
            break
        x = x / nrm
    eig = np.abs(x) * fg

    m = np.zeros((M, N), np.float32)
    for j in range(N):
        col = img_of == j
        m[:, j] = sim[:, col].max(1) if col.any() else 0.0
    cov = np.zeros(M, np.float32)
    for p in range(M):
        others = np.delete(m[p], img_of[p])
        k = max(1, int(np.ceil(low_frac * len(others))))
        cov[p] = np.sort(others)[:k].mean()
    cov = cov * fg

    score = _unit(eig) * (_unit(cov) ** gamma)
    return score, img_of, club_of


def consensus_assignment(data, key="club_super_embeddings",
                         low_frac=0.5, gamma=1.0, fg_pow=1.0, iters=100, rounds=20):
    """
    Coordinate-ascent consensus assignment (general, no category constants).
      init  : per-image argmax of the set-based score (fg x eig x cov).
      update: a_i <- argmax_{c in image i}  sum_{j!=i} sim(c, a_j) * fg(c)
              i.e. the club that best AGREES with the objects the other images
              currently point to. A distractor that matches different random
              targets per image gets no coherent agreement; the recurring object
              (whose partners all match each other) is a strong attractor.
    Returns per-image (club_id, pixel_mask).
    """
    sim, img_of, club_of = set_similarity(data, key)
    N = int(img_of.max()) + 1
    fg = foreground_prior(data) ** fg_pow

    score, _, _ = set_cosaliency(data, key, low_frac, gamma, fg_pow, iters)
    a = np.zeros(N, int)                                  # global club index chosen per image
    for n in range(N):
        sel = np.where(img_of == n)[0]
        a[n] = sel[np.argmax(score[sel])]

    s_unit = _unit(score)                                # independent evidence, in [0,1]
    for _ in range(rounds):
        changed = False
        for i in range(N):
            sel = np.where(img_of == i)[0]
            others = a[np.arange(N) != i]
            agree = sim[np.ix_(sel, others)].mean(1)           # agreement with others' picks
            # evidence x consensus: a club must be clock-like AND agree with the group
            cand = _unit(agree) * s_unit[sel]
            best = sel[int(np.argmax(cand))]
            if best != a[i]:
                a[i] = best; changed = True
        if not changed:
            break

    out = []
    for n in range(N):
        cid = int(club_of[a[n]])
        out.append((cid, data[n]["label_map"] == cid))
    return out


def select(score, img_of, club_of, data):
    N = int(img_of.max()) + 1
    out = []
    for n in range(N):
        sel = np.where(img_of == n)[0]
        cid = int(club_of[sel[np.argmax(score[sel])]])
        out.append((cid, data[n]["label_map"] == cid))
    return out
