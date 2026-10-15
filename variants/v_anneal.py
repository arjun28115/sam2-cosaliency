"""
Annealed-consensus variant (final method candidate).
Stage 1: baseline evidence + gated assignment (gate=0.5) -> good initial picks.
Stage 2: refinement rounds with PURE group agreement (gate=0), where agreement
         uses QUANTILE-aggregated directional Chamfer: for the direction from
         another image's pick into this candidate, aggregate over the source
         parts with a low quantile q — ALL parts of the group's object
         (incl. its distinctive face) must find a match in the candidate.
         A mug matches dark clock bodies but not their faces -> its
         worst-source-parts term collapses. Fully general.
Stage 3: marginal-gain prune of the picked unions (drop dead-weight members).
"""
import numpy as np
from harness import candidates, rbd_fg_candidates
from v_base import evidence, gated_assignment, _unit
from v_combo import _prep, _cache, weighted_set_sim

_local = {}


def _sim_base(data, max_union=3):
    U, cr, ci, cm, sp_img, fg = _prep(data, max_union)
    key = ("sim", 0.0, 0.1)
    if key not in _cache:
        w = np.ones(len(U), np.float32)
        _cache[key] = weighted_set_sim(U, cr, ci, w)
    sim, maxub, G = _cache[key]
    return U, cr, ci, cm, fg, sim, G


def _qdir(G, rows_src, rows_dst, q):
    """quantile-aggregated directional Chamfer src->dst (q=1.0 -> mean)."""
    best = G[np.ix_(rows_src, rows_dst)].max(1)
    if q >= 1.0:
        return float(best.mean())
    k = max(1, int(np.ceil(q * len(best))))
    return float(np.sort(best)[:k].mean())        # mean of the worst q-fraction


def _qsim(G, rows_a, rows_b, q):
    return min(_qdir(G, rows_a, rows_b, q), _qdir(G, rows_b, rows_a, q))


def refine(G, cr, ci, a, q=0.3, rounds=8):
    """Pure-agreement reassignment with quantile Chamfer to others' picks."""
    N = int(ci.max()) + 1
    sel_by_img = [np.where(ci == n)[0] for n in range(N)]
    for _ in range(rounds):
        changed = False
        for i in range(N):
            others = [a[j] for j in range(N) if j != i]
            vals = np.array([np.mean([_qsim(G, cr[k], cr[o], q) for o in others])
                             for k in sel_by_img[i]])
            b = sel_by_img[i][int(np.argmax(vals))]
            if b != a[i]:
                a[i] = b; changed = True
        if not changed:
            break
    return a


def prune(G, cr, ci, cm, a, q=0.3):
    """Greedy backward member selection maximizing quantile agreement."""
    N = int(ci.max()) + 1
    rows_of_club = {}
    for k, mem in enumerate(cm):
        if len(mem) == 1:
            rows_of_club[(ci[k], next(iter(mem)))] = cr[k]
    out = []
    for n in range(N):
        others = [a[j] for j in range(N) if j != n]
        mem = sorted(cm[a[n]])

        def agree(members):
            rows = np.concatenate([rows_of_club[(n, c)] for c in members])
            return float(np.mean([_qsim(G, rows, cr[o], q) for o in others]))

        cur = agree(mem)
        while len(mem) > 1:
            best_drop, best_val = None, cur
            for c in mem:
                trial = [x for x in mem if x != c]
                v = agree(trial)
                if v >= best_val:
                    best_val, best_drop = v, c
            if best_drop is None:
                break
            mem.remove(best_drop); cur = best_val
        out.append(mem)
    return out


def run(data, q=0.3, refine_rounds=8, do_prune=True,
        max_union=3, low_frac=0.35, gamma=1.0, gate=0.5):
    U, cr, ci, cm, fg, sim, G = _sim_base(data, max_union)
    ev = evidence(sim, ci, fg, low_frac, gamma)
    a = gated_assignment(sim, ci, ev, gate)          # stage 1
    a = refine(G, cr, ci, a.copy(), q, refine_rounds)  # stage 2
    if do_prune:
        mems = prune(G, cr, ci, cm, a, q)            # stage 3
    else:
        mems = [sorted(cm[a[n]]) for n in range(int(ci.max()) + 1)]
    picks = []
    for n, mem in enumerate(mems):
        lm = data[n]["label_map"]
        picks.append((sorted(mem), np.isin(lm, list(mem))))
    return picks


if __name__ == "__main__":
    from harness import load, sweep, save_results
    data = load()
    res = sweep(run, {"q": [1.0, 0.5, 0.3, 0.2], "do_prune": [True, False]},
                data, tag="anneal")
    save_results(res, "/Users/arjunsingla/sam2/variants/results_anneal.json")
