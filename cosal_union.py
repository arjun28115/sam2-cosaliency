"""
Union-candidate co-saliency  —  SAM2-only, general.
===================================================
Root cause of the last failures: a small object is split by segmentation into
parts (a white clock FACE club + a dark RIM/BELLS club). The isolated white
face is a bare disc that min-Chamfer cannot tell from a white blanket / mug.

Fix at the CANDIDATE level (no scoring hack, no category knowledge):
  candidate(i) in { single club } U { connected unions of <=K adjacent clubs }.
A reunited object (face+rim+bells) carries its full multi-part signature, which
only matches other whole objects -> beats uniform distractors. Everything else
(min-Chamfer set similarity, fg, eig, cov, consensus assignment) is unchanged;
it just runs over the richer candidate set.
"""
import numpy as np
from itertools import combinations
from cosal_spectral import foreground_prior
from cosal_set import _unit


def _club_adjacency(label_map, club_ids):
    """Adjacency (shared boundary) among the given club ids in one image."""
    lm = label_map
    adj = {c: set() for c in club_ids}
    idx = {c: i for i, c in enumerate(club_ids)}
    # horizontal & vertical neighbour pairs
    for A, B in ((lm[:, :-1], lm[:, 1:]), (lm[:-1, :], lm[1:, :])):
        d = A != B
        for a, b in zip(A[d].ravel(), B[d].ravel()):
            a, b = int(a), int(b)
            if a in adj and b in adj:
                adj[a].add(b); adj[b].add(a)
    return adj


def _candidates(data, key="club_super_embeddings", max_union=3):
    """
    Build multi-granularity candidates per image.
    Returns:
      U (T,256) L2-normalised superpixel embeddings (all images stacked),
      cand_rows : list of np.array row-indices into U (one per candidate),
      cand_img  : (Ncand,) image index,
      cand_members : list of frozenset(club_ids),
      data_clubsp : per-image {club_id: [global rows]} for mask building.
    """
    U, sp_img = [], []
    cand_rows, cand_img, cand_members = [], [], []
    t = 0
    for n, g in enumerate(data):
        se = np.asarray(g["super_embeddings"], np.float32)
        se = se / (np.linalg.norm(se, axis=1, keepdims=True) + 1e-8)
        sp_ids = list(np.asarray(g["superpixel_ids"])); sp2c = g["sp_to_club"]
        by_club = {}
        for k, spid in enumerate(sp_ids):
            cid = int(sp2c[int(spid)])
            by_club.setdefault(cid, []).append(t + k)
        U.append(se); t += len(sp_ids)

        club_ids = [int(c) for c in g[key].keys() if int(c) in by_club]
        rows_of = {c: np.array(by_club[c]) for c in club_ids}
        adj = _club_adjacency(g["label_map"], club_ids)

        # connected subsets of size 1..max_union (BFS-limited enumeration)
        seen = set()
        def add(members):
            fs = frozenset(members)
            if fs in seen:
                return
            seen.add(fs)
            rows = np.concatenate([rows_of[c] for c in members])
            cand_rows.append(rows); cand_img.append(n); cand_members.append(fs)
        for c in club_ids:
            add([c])
        # grow connected sets
        frontier = [frozenset([c]) for c in club_ids]
        for _ in range(max_union - 1):
            newf = []
            for s in frontier:
                nbrs = set().union(*[adj[c] for c in s]) - s
                for c in nbrs:
                    ns = s | {c}
                    if ns not in seen:
                        add(ns); newf.append(ns)
            frontier = newf
    U = np.concatenate(U, 0)
    return U, cand_rows, np.array(cand_img), cand_members


def _set_sim_from_candidates(U, cand_rows, cand_img):
    G = np.maximum(U @ U.T, 0.0)
    C = len(cand_rows); T = len(U)
    maxub = np.empty((T, C), np.float32)
    for b in range(C):
        maxub[:, b] = G[:, cand_rows[b]].max(1)
    dirm = np.empty((C, C), np.float32)
    for a in range(C):
        dirm[a] = maxub[cand_rows[a]].mean(0)
    sim = np.minimum(dirm, dirm.T)
    same = cand_img[:, None] == cand_img[None, :]
    sim[same] = 0.0
    return sim


def _fg_for_candidates(data, cand_img, cand_members, key="club_super_embeddings"):
    """RBD foreground weight for a candidate = area-weighted mean over its member clubs."""
    fg_club = foreground_prior(data)
    # map (img, club_id) -> flat club index used by foreground_prior (same iteration order)
    idx = {}; t = 0
    for n, g in enumerate(data):
        for cid in g[key].keys():
            idx[(n, int(cid))] = t; t += 1
    areas = {}
    for n, g in enumerate(data):
        lm = g["label_map"]
        for cid in g[key].keys():
            areas[(n, int(cid))] = float((lm == int(cid)).sum())
    out = np.empty(len(cand_img), np.float32)
    for k, (n, mem) in enumerate(zip(cand_img, cand_members)):
        w = np.array([areas[(n, c)] for c in mem]); f = np.array([fg_club[idx[(n, c)]] for c in mem])
        out[k] = (w * f).sum() / (w.sum() + 1e-8)
    return out


def union_cosaliency(data, key="club_super_embeddings",
                     low_frac=0.35, gamma=1.0, fg_pow=1.0, iters=100,
                     max_union=3, rounds=25, gate=1.0, w_eig=1.0):
    U, cand_rows, cand_img, cand_members = _candidates(data, key, max_union)
    sim = _set_sim_from_candidates(U, cand_rows, cand_img)
    fg = _fg_for_candidates(data, cand_img, cand_members, key) ** fg_pow
    N = int(cand_img.max()) + 1; M = len(sim)

    # eig (mutual consistency)
    A = (fg[:, None] * sim) * fg[None, :]
    x = fg.copy() + 1e-6
    for _ in range(iters):
        x = A @ x; nrm = np.linalg.norm(x)
        if nrm < 1e-12: break
        x = x / nrm
    eig = np.abs(x) * fg
    # cov (ubiquity)
    m = np.zeros((M, N), np.float32)
    for j in range(N):
        col = cand_img == j
        m[:, j] = sim[:, col].max(1) if col.any() else 0.0
    cov = np.zeros(M, np.float32)
    for p in range(M):
        others = np.delete(m[p], cand_img[p]); k = max(1, int(np.ceil(low_frac * len(others))))
        cov[p] = np.sort(others)[:k].mean()
    cov *= fg
    score = (_unit(eig) ** w_eig) * (_unit(cov) ** gamma)

    # init assignment: best candidate per image
    a = np.zeros(N, int)
    for n in range(N):
        sel = np.where(cand_img == n)[0]; a[n] = sel[np.argmax(score[sel])]
    # gated consensus assignment (evidence x agreement)
    s_unit = _unit(score)
    for _ in range(rounds):
        changed = False
        for i in range(N):
            sel = np.where(cand_img == i)[0]; others = a[np.arange(N) != i]
            agree = sim[np.ix_(sel, others)].mean(1)
            # evidence x agreement; gate<1 lets group agreement drive ambiguous ties
            cand = _unit(agree) * (s_unit[sel] ** gate)
            best = sel[int(np.argmax(cand))]
            if best != a[i]:
                a[i] = best; changed = True
        if not changed: break

    # build masks from chosen candidate member clubs
    out = _masks(data, cand_members, a, N, cand_img)
    return out


def global_assignment(data, key="club_super_embeddings",
                      low_frac=0.35, gamma=1.0, fg_pow=1.0, iters=100,
                      max_union=3, rounds=30, lam=0.15, restarts=12):
    """
    Global consensus clique: pick one candidate per image maximizing
        J(a) = sum_{i<j} sim(a_i, a_j)  +  lam * sum_i evidence(a_i).
    The recurring object forms the maximum-agreement clique across all N images;
    a distractor present in one image cannot. Coordinate ascent finds a local
    optimum, so run it from many inits and keep the highest-J assignment.
    """
    U, cand_rows, cand_img, cand_members = _candidates(data, key, max_union)
    sim = _set_sim_from_candidates(U, cand_rows, cand_img)
    fg = _fg_for_candidates(data, cand_img, cand_members, key) ** fg_pow
    N = int(cand_img.max()) + 1; M = len(sim)

    A = (fg[:, None] * sim) * fg[None, :]
    x = fg.copy() + 1e-6
    for _ in range(iters):
        x = A @ x; nrm = np.linalg.norm(x)
        if nrm < 1e-12: break
        x = x / nrm
    eig = np.abs(x) * fg
    m = np.zeros((M, N), np.float32)
    for j in range(N):
        col = cand_img == j
        m[:, j] = sim[:, col].max(1) if col.any() else 0.0
    cov = np.zeros(M, np.float32)
    for p in range(M):
        o = np.delete(m[p], cand_img[p]); k = max(1, int(np.ceil(low_frac * len(o))))
        cov[p] = np.sort(o)[:k].mean()
    cov *= fg
    ev = _unit(eig) * (_unit(cov) ** gamma)               # per-candidate evidence
    sel_by_img = [np.where(cand_img == n)[0] for n in range(N)]

    def objective(a):
        s = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                s += sim[a[i], a[j]]
        return s + lam * ev[a].sum()

    def ascend(a):
        for _ in range(rounds):
            changed = False
            for i in range(N):
                others = a[np.arange(N) != i]
                val = sim[np.ix_(sel_by_img[i], others)].sum(1) + lam * ev[sel_by_img[i]]
                b = sel_by_img[i][int(np.argmax(val))]
                if b != a[i]:
                    a[i] = b; changed = True
            if not changed: break
        return a

    best_a, best_J = None, -1e18
    for r in range(restarts):
        a = np.array([sel_by_img[n][np.argsort(-ev[sel_by_img[n]])[min(r, len(sel_by_img[n]) - 1)]]
                      for n in range(N)])              # init: r-th best-evidence candidate per image
        a = ascend(a); J = objective(a)
        if J > best_J:
            best_J, best_a = J, a.copy()
    return _masks(data, cand_members, best_a, N, cand_img)


def _masks(data, cand_members, a, N, cand_img):
    out = []
    for n in range(N):
        mem = cand_members[a[n]]; lm = data[n]["label_map"]
        mask = np.isin(lm, list(mem))
        out.append((sorted(mem), mask))
    return out
