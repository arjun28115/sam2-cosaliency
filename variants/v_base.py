"""
Baseline variant: union candidates + min-Chamfer set similarity
               + RBD foreground x eigenvector x robust-coverage evidence
               + gated consensus assignment.
(The 15/16 method from the grey data, ported to the shared harness.)
"""
import numpy as np
from harness import candidates, rbd_fg_candidates


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (v.max() + 1e-12)


def set_sim(U, cand_rows, cand_img):
    """Symmetric min-Chamfer similarity between candidates; same-image zeroed."""
    G = np.maximum(U @ U.T, 0.0)
    C = len(cand_rows); T = len(U)
    maxub = np.empty((T, C), np.float32)
    for b in range(C):
        maxub[:, b] = G[:, cand_rows[b]].max(1)
    dirm = np.empty((C, C), np.float32)
    for a in range(C):
        dirm[a] = maxub[cand_rows[a]].mean(0)
    sim = np.minimum(dirm, dirm.T)
    sim[cand_img[:, None] == cand_img[None, :]] = 0.0
    return sim


def evidence(sim, cand_img, fg, low_frac=0.35, gamma=1.0, iters=100):
    """fg x eigenvector(mutual consistency) x robust coverage(ubiquity)."""
    N = int(cand_img.max()) + 1; M = len(sim)
    A = (fg[:, None] * sim) * fg[None, :]
    x = fg.copy() + 1e-6
    for _ in range(iters):
        x = A @ x; nrm = np.linalg.norm(x)
        if nrm < 1e-12:
            break
        x = x / nrm
    eig = np.abs(x) * fg
    m = np.zeros((M, N), np.float32)
    for j in range(N):
        col = cand_img == j
        m[:, j] = sim[:, col].max(1) if col.any() else 0.0
    cov = np.zeros(M, np.float32)
    for p in range(M):
        o = np.delete(m[p], cand_img[p])
        k = max(1, int(np.ceil(low_frac * len(o))))
        cov[p] = np.sort(o)[:k].mean()
    cov *= fg
    return _unit(eig) * (_unit(cov) ** gamma)


def gated_assignment(sim, cand_img, ev, gate=0.5, rounds=30):
    N = int(cand_img.max()) + 1
    sel_by_img = [np.where(cand_img == n)[0] for n in range(N)]
    a = np.array([s[np.argmax(ev[s])] for s in sel_by_img])
    s_unit = _unit(ev)
    for _ in range(rounds):
        changed = False
        for i in range(N):
            others = a[np.arange(N) != i]
            agree = sim[np.ix_(sel_by_img[i], others)].mean(1)
            val = _unit(agree) * (s_unit[sel_by_img[i]] ** gate)
            b = sel_by_img[i][int(np.argmax(val))]
            if b != a[i]:
                a[i] = b; changed = True
        if not changed:
            break
    return a


def masks_from(data, cand_members, cand_img, a):
    N = int(cand_img.max()) + 1
    out = []
    for n in range(N):
        mem = cand_members[a[n]]; lm = data[n]["label_map"]
        out.append((sorted(mem), np.isin(lm, list(mem))))
    return out


def run(data, max_union=3, low_frac=0.35, gamma=1.0, gate=0.5):
    U, cr, ci, cm = candidates(data, max_union)
    sim = set_sim(U, cr, ci)
    fg = rbd_fg_candidates(data, ci, cm)
    ev = evidence(sim, ci, fg, low_frac, gamma)
    a = gated_assignment(sim, ci, ev, gate)
    return masks_from(data, cm, ci, a)


if __name__ == "__main__":
    from harness import load, sweep, save_results
    data = load()
    res = sweep(run, {"max_union": [3], "low_frac": [0.35, 0.5],
                      "gamma": [1.0], "gate": [0.5, 1.0]}, data, tag="base")
    save_results(res, "/Users/arjunsingla/sam2/variants/results_base.json")
