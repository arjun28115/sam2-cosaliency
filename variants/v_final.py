"""
FINAL METHOD — SAM2-only, training-free, general co-saliency.
=============================================================
Pipeline (all category-agnostic):
 1. Candidates: clubs + connected unions (<=3) from the SAM2 partition,
    orphan-repaired label maps.
 2. Stage-1 scoring: min-Chamfer set similarity on superpixel embeddings;
    evidence = RBD-foreground x eigenvector(mutual consistency) x robust
    coverage; gated consensus assignment -> initial picks.
 3. Finalists: per image, top-K by evidence U top-K by agreement.
 4. Re-encode every finalist with SAM2 at crop scale (raw + masked variants)
    -> object-centric 256-d descriptors.
 5. Consensus clique over finalists with the soft-AND pair similarity
       pair(a,b) = nrm(qsim) * nrm(cos_raw) * nrm(cos_masked)
    (each view kills a different distractor family: patch-level Chamfer kills
    texture-frankensteins; raw crops kill wrong whole-object shapes; masked
    crops rescue small objects from context pollution)
 6. Marginal-gain prune of the picked unions.
"""
import numpy as np
import sys
sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
from harness import candidates, rbd_fg_candidates
from v_base import evidence, gated_assignment, _unit
from v_combo import _prep, _cache, weighted_set_sim
from v_anneal import _qsim
import v_reencode as vr


def _nrm(M):
    lo, hi = M.min(), M.max()
    return (M - lo) / (hi - lo + 1e-12)


def _rank(M, valid):
    """Empirical-CDF transform over valid (cross-image) pairs: scale-invariant,
    robust to finalist-pool composition (min-max compresses when new high-sim
    pairs enter the pool, silently re-weighting lambda*evidence)."""
    out = np.zeros_like(M, dtype=np.float32)
    vals = M[valid]
    if len(vals) == 0:
        return out
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(len(vals), np.float32)
    ranks[order] = np.arange(1, len(vals) + 1, dtype=np.float32) / len(vals)
    out[valid] = ranks
    return out


def run(data, K=10, q=0.3, lam=0.15, use_raw=True, use_masked=True,
        use_qsim=True, do_prune=True, restarts=6, small_k=0):
    short, (U, cr, ci, cm, fg, sim, ev, a0) = vr._shortlist(data, K, small_k=small_k)
    d_raw = vr._descriptors(data, short, ci, cm, masked=False) if use_raw else None
    d_msk = vr._descriptors(data, short, ci, cm, masked=True) if use_masked else None
    G = np.maximum(U @ U.T, 0.0)
    N = len(short)
    allk = [k for s in short for k in s]
    idx = {k: i for i, k in enumerate(allk)}
    M = len(allk)

    # pairwise signal matrices over finalists
    S_q = np.zeros((M, M), np.float32)
    S_r = np.zeros((M, M), np.float32)
    S_m = np.zeros((M, M), np.float32)
    for i, a in enumerate(allk):
        for jj, b in enumerate(allk):
            if jj <= i or ci[a] == ci[b]:
                continue
            if use_qsim:
                S_q[i, jj] = S_q[jj, i] = _qsim(G, cr[a], cr[b], q)
            if use_raw:
                S_r[i, jj] = S_r[jj, i] = max(0.0, float(d_raw[a] @ d_raw[b]))
            if use_masked:
                S_m[i, jj] = S_m[jj, i] = max(0.0, float(d_msk[a] @ d_msk[b]))
    same = np.array([[ci[a] == ci[b] for b in allk] for a in allk])
    valid = ~same & ~np.eye(M, dtype=bool)
    P = np.ones((M, M), np.float32)
    if use_qsim:
        P = P * _rank(S_q, valid)
    if use_raw:
        P = P * _rank(S_r, valid)
    if use_masked:
        P = P * _rank(S_m, valid)
    P[same] = 0.0

    # co-SALIENCY = recurring AND salient: without foreground weighting the
    # clique objective prefers the most self-similar recurring region, which
    # in scenic groups is the sky/background (night skies are near-identical).
    fgk = np.array([fg[k] for k in allk], np.float32)
    # gamma=0.5: strong enough to kill sky/background cliques (fg ~ 0.001),
    # soft enough to spare true objects clipped by the frame (fg ~ 0.3-0.5)
    P = P * (np.outer(fgk, fgk) ** 0.25)

    evu = _unit(ev)

    def ascend(a):
        for _ in range(20):
            changed = False
            for i in range(N):
                others = [a[j] for j in range(N) if j != i]
                vals = [np.mean([P[idx[k], idx[o]] for o in others]) + lam * evu[k]
                        for k in short[i]]
                b = short[i][int(np.argmax(vals))]
                if b != a[i]:
                    a[i] = b; changed = True
            if not changed:
                break
        return a

    def objective(a):
        s = sum(P[idx[a[i]], idx[a[j]]] for i in range(N) for j in range(i + 1, N))
        return s + lam * sum(evu[a[i]] for i in range(N))

    inits = [[a0[n] if a0[n] in short[n] else short[n][0] for n in range(N)]]
    for r in range(restarts - 1):
        inits.append([short[n][np.argsort([-evu[k] for k in short[n]])[min(r, len(short[n]) - 1)]]
                      for n in range(N)])
    # pair-seeded greedy inits: evidence-ranked inits all start in the same
    # basin (big salient regions); when the true object's clique is strong but
    # its members have LOW evidence (a small moon), coordinate ascent can never
    # escape — a single flip gains no pair-sim. Seed instead from the strongest
    # cross-image pairs and grow by pair-sim to the seed set.
    Pf = P.copy()
    used_pairs = set()
    for _ in range(3):
        flat = np.argmax(Pf)
        i0, j0 = np.unravel_index(flat, Pf.shape)
        if Pf[i0, j0] <= 0:
            break
        na, nb = ci[allk[i0]], ci[allk[j0]]
        if (na, nb) in used_pairs:
            Pf[i0, j0] = Pf[j0, i0] = 0
            continue
        used_pairs.add((na, nb)); used_pairs.add((nb, na))
        seed = {na: allk[i0], nb: allk[j0]}
        for n in sorted(range(N), key=lambda x: (x in seed)):
            if n in seed:
                continue
            svals = [np.mean([P[idx[k], idx[s]] for s in seed.values()]) for k in short[n]]
            seed[n] = short[n][int(np.argmax(svals))]
        inits.append([seed[n] for n in range(N)])
        Pf[i0, j0] = Pf[j0, i0] = 0
    best_a, best_J = None, -1e18
    for a in inits:
        a = ascend(list(a)); J = objective(a)
        if J > best_J:
            best_J, best_a = J, list(a)

    # marginal-gain prune + greedy expansion with the qsim signal
    mems = []
    if do_prune:
        from harness import _club_adjacency
        rows_of_club = {}
        for k, mem in enumerate(cm):
            if len(mem) == 1:
                rows_of_club[(ci[k], next(iter(mem)))] = cr[k]
        for n in range(N):
            others = [best_a[j] for j in range(N) if j != n]
            mem = sorted(cm[best_a[n]])
            usable = [c for c in np.unique(data[n]["label_map"]) if (n, int(c)) in rows_of_club]
            adj = _club_adjacency(data[n]["label_map"], [int(c) for c in usable])

            def agree(members):
                rows = np.concatenate([rows_of_club[(n, c)] for c in members])
                return float(np.mean([_qsim(G, rows, cr[o], q) for o in others]))

            cur = agree(mem)
            # backward: drop dead-weight members
            while len(mem) > 1:
                bd, bv = None, cur
                for c in mem:
                    t = [x for x in mem if x != c]
                    v = agree(t)
                    if v >= bv:
                        bv, bd = v, c
                if bd is None:
                    break
                mem.remove(bd); cur = bv
            # forward: absorb adjacent clubs, gated by the SAME soft-AND signal
            # as the clique — the expanded object's masked-crop re-encode must
            # also agree better with the group (a clock+lamp-pole crop looks
            # less clock-like; a completed clover looks more clover-like).
            if use_masked:
                from PIL import Image as _I
                img_full = np.array(_I.open(data[n]["meta"]["file"]).convert("RGB"))
                lm_full = np.array(_I.fromarray(data[n]["label_map"].astype(np.int32), mode="I")
                                   .resize((img_full.shape[1], img_full.shape[0]), _I.NEAREST))
                o_desc = [d_msk[o] for o in others]

                def crop_agree(members):
                    """top-3 (not mean): parts shared by only a SUBSET of the
                    group (dark clover centers) must still be absorbable —
                    mean-consensus drops them, peak-consensus keeps them."""
                    mfull = np.isin(lm_full, members)
                    if not mfull.any():
                        return 0.0
                    v = vr._encode_crop(img_full, mfull, masked=True)
                    vals = sorted((max(0.0, v @ od) for od in o_desc), reverse=True)
                    return float(np.mean(vals[:3]))

                cur_c = crop_agree(mem)
                while True:
                    nbrs = sorted(set().union(*[adj.get(c, set()) for c in mem]) - set(mem))
                    ba, bv, bc = None, cur, cur_c
                    for c in nbrs:
                        v = agree(mem + [c])
                        if v < cur:                       # patch-level gate first (cheap)
                            continue
                        vc = crop_agree(mem + [c])        # object-level gate (one encode)
                        if vc > bc:
                            bv, bc, ba = v, vc, c
                    if ba is None:
                        break
                    mem.append(ba); cur, cur_c = bv, bc
            mems.append(sorted(mem))
    else:
        mems = [sorted(cm[best_a[n]]) for n in range(N)]

    picks = []
    for n, mem in enumerate(mems):
        lm = data[n]["label_map"]
        picks.append((sorted(mem), np.isin(lm, mem)))
    return picks


if __name__ == "__main__":
    from harness import load, evaluate, save_results
    data = load()
    results = []
    for cfg in [dict(use_qsim=True, use_raw=True, use_masked=True),
                dict(use_qsim=True, use_raw=True, use_masked=False),
                dict(use_qsim=True, use_raw=False, use_masked=True),
                dict(use_qsim=False, use_raw=True, use_masked=True)]:
        picks = run(data, **cfg)
        hits, misses, det = evaluate(picks, data)
        print(f"[final] {cfg} -> {len(hits)}/16 miss={misses}", flush=True)
        results.append({"params": cfg, "hits": len(hits), "misses": misses, "detail": det})
    results.sort(key=lambda r: -r["hits"])
    save_results(results, "/Users/arjunsingla/sam2/variants/results_final.json")
