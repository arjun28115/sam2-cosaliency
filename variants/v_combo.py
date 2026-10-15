"""
Combo variant: three orthogonal, general mechanisms on top of the baseline.

  tfidf     : cross-image genericness weighting of superpixels inside the
              Chamfer. genericness(u) = mean over other images of the mean of
              the top-p fraction of cos(u, v in j)  -> captures MULTIPLICITY
              (generic texture matches many superpixels everywhere; a
              distinctive object part matches few). w_u = (1 - rank(g_u))^beta.
  coherence : union candidates whose member clubs are mutually dissimilar
              (frankenstein unions assembled from different objects) are
              penalised: evidence *= (relu(min pairwise club cosine))^ce.
  prune     : after the gated assignment, greedily drop members of the picked
              union while doing so does not reduce its agreement with the other
              images' picks (dead-weight members like a vase merged with a
              clock face).

All scale-free (rank/percentile-based), no category knowledge.
"""
import numpy as np
from harness import candidates, rbd_fg_candidates
from v_base import evidence, gated_assignment, masks_from, _unit

_cache = {}


def _prep(data, max_union=3):
    if "prep" not in _cache:
        U, cr, ci, cm = candidates(data, max_union)
        sp_img = np.empty(len(U), np.int32)
        for k, rows in enumerate(cr):
            sp_img[rows] = ci[k]          # every row appears in its image's singletons
        fg = rbd_fg_candidates(data, ci, cm)
        _cache["prep"] = (U, cr, ci, cm, sp_img, fg)
    return _cache["prep"]


def genericness(U, sp_img, top_p=0.1):
    key = ("gen", top_p)
    if key not in _cache:
        T = len(U); N = int(sp_img.max()) + 1
        g = np.zeros(T, np.float32); cnt = np.zeros(T, np.float32)
        for j in range(N):
            cols = np.where(sp_img == j)[0]
            C = np.maximum(U @ U[cols].T, 0.0)              # (T, Tj)
            k = max(1, int(np.ceil(top_p * len(cols))))
            part = np.partition(C, -k, axis=1)[:, -k:]      # top-k per row
            gj = part.mean(1)
            mask = sp_img != j
            g[mask] += gj[mask]; cnt[mask] += 1
        g = g / np.maximum(cnt, 1)
        # rank-normalise to [0,1] (scale-free)
        order = np.argsort(g)
        r = np.empty(T, np.float32); r[order] = np.arange(T) / max(1, T - 1)
        _cache[key] = r
    return _cache[key]


def weighted_set_sim(U, cr, ci, w):
    """min-Chamfer with per-superpixel weights w in the source direction."""
    G = np.maximum(U @ U.T, 0.0)
    C = len(cr); T = len(U)
    maxub = np.empty((T, C), np.float32)
    for b in range(C):
        maxub[:, b] = G[:, cr[b]].max(1)
    dirm = np.empty((C, C), np.float32)
    for a in range(C):
        rows = cr[a]; wa = w[rows]
        dirm[a] = (wa[:, None] * maxub[rows]).sum(0) / (wa.sum() + 1e-8)
    sim = np.minimum(dirm, dirm.T)
    sim[ci[:, None] == ci[None, :]] = 0.0
    return sim, maxub, G


def union_coherence(U, cr, ci, cm, data):
    """min pairwise cosine between member-club mean embeddings (1 for singletons)."""
    if "coh" in _cache:
        return _cache["coh"]
    # club mean embeddings from rows of the singleton candidates
    club_mean = {}
    for k, mem in enumerate(cm):
        if len(mem) == 1:
            c = next(iter(mem))
            v = U[cr[k]].mean(0)
            club_mean[(ci[k], c)] = v / (np.linalg.norm(v) + 1e-8)
    coh = np.ones(len(cm), np.float32)
    for k, mem in enumerate(cm):
        if len(mem) > 1:
            ms = [club_mean[(ci[k], c)] for c in mem]
            vals = [float(ms[i] @ ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
            coh[k] = max(0.0, min(vals))
    _cache["coh"] = coh
    return coh


def _dir_w(U, rows_a, rows_b, w):
    """weighted directional Chamfer a->b on the fly."""
    C = np.maximum(U[rows_a] @ U[rows_b].T, 0.0)
    wa = w[rows_a]
    return float((wa * C.max(1)).sum() / (wa.sum() + 1e-8))


def _sim_pair(U, rows_a, rows_b, w):
    return min(_dir_w(U, rows_a, rows_b, w), _dir_w(U, rows_b, rows_a, w))


def prune_pick(U, w, data, cm, cr, ci, a, n, rows_of_club):
    """Greedy backward selection on image n's picked union: drop members while
    agreement with the other images' picks does not decrease."""
    N = int(ci.max()) + 1
    others = [a[j] for j in range(N) if j != n]
    mem = sorted(cm[a[n]])
    if len(mem) == 1:
        return mem

    def agree(members):
        rows = np.concatenate([rows_of_club[(n, c)] for c in members])
        return float(np.mean([_sim_pair(U, rows, cr[o], w) for o in others]))

    cur = agree(mem)
    improved = True
    while improved and len(mem) > 1:
        improved = False
        best_drop, best_val = None, cur
        for c in mem:
            trial = [x for x in mem if x != c]
            v = agree(trial)
            if v >= best_val:                      # dropping c does not hurt
                best_val, best_drop = v, c
        if best_drop is not None:
            mem.remove(best_drop); cur = best_val; improved = True
    return mem


def run(data, beta=1.0, top_p=0.1, ce=0.0, prune=False,
        max_union=3, low_frac=0.35, gamma=1.0, gate=0.5):
    U, cr, ci, cm, sp_img, fg = _prep(data, max_union)
    if beta > 0:
        g = genericness(U, sp_img, top_p)
        w = (1.0 - g) ** beta
    else:
        w = np.ones(len(U), np.float32)
    key = ("sim", beta, top_p)
    if key not in _cache:
        _cache[key] = weighted_set_sim(U, cr, ci, w)
    sim, maxub, G = _cache[key]

    ev = evidence(sim, ci, fg, low_frac, gamma)
    if ce > 0:
        coh = union_coherence(U, cr, ci, cm, data)
        ev = ev * (coh ** ce)
    a = gated_assignment(sim, ci, ev, gate)

    if prune:
        rows_of_club = {}
        for k, mem in enumerate(cm):
            if len(mem) == 1:
                rows_of_club[(ci[k], next(iter(mem)))] = cr[k]
        picks = []
        N = int(ci.max()) + 1
        for n in range(N):
            mem = prune_pick(U, w, data, cm, cr, ci, a, n, rows_of_club)
            lm = data[n]["label_map"]
            picks.append((sorted(mem), np.isin(lm, list(mem))))
        return picks
    return masks_from(data, cm, ci, a)


if __name__ == "__main__":
    import sys
    from harness import load, sweep, save_results
    data = load()
    grid = {"beta": [0.0, 1.0, 2.0], "ce": [0.0, 0.5], "prune": [False, True]}
    res = sweep(run, grid, data, tag="combo")
    save_results(res, "/Users/arjunsingla/sam2/variants/results_combo.json")
