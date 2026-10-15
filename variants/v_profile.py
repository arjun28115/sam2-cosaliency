"""
Second-order (profile) similarity variant.
Two candidates are the same object iff they MATCH THE SAME THINGS across the
group: profile(a) = row of the base min-Chamfer similarity S over candidates in
OTHER images; S2[a,b] = Pearson/cosine correlation of profiles. A distractor
(teddy) correlates with soft-texture stuff; clocks correlate with clocks.
Suse = alpha*S + (1-alpha)*unit(S2), then baseline evidence + gated assignment.
"""
import numpy as np
from harness import candidates, rbd_fg_candidates
from v_base import set_sim, evidence, gated_assignment, masks_from

_cache = {}


def _base(data, max_union=3):
    key = ("base", max_union)
    if key not in _cache:
        U, cr, ci, cm = candidates(data, max_union)
        S = set_sim(U, cr, ci)
        fg = rbd_fg_candidates(data, ci, cm)
        _cache[key] = (U, cr, ci, cm, S, fg)
    return _cache[key]


def second_order(S, ci, kind="pearson"):
    key = ("s2", kind)
    if key in _cache:
        return _cache[key]
    M = len(S)
    # profile rows: same-image entries already zeroed in S
    P = S.astype(np.float32).copy()
    if kind == "pearson":
        P = P - P.mean(1, keepdims=True)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S2 = Pn @ Pn.T
    S2 = np.maximum(S2, 0.0)
    S2[ci[:, None] == ci[None, :]] = 0.0
    _cache[key] = S2
    return S2


def run(data, alpha=0.5, kind="pearson", max_union=3, low_frac=0.35, gamma=1.0, gate=0.5):
    U, cr, ci, cm, S, fg = _base(data, max_union)
    S2 = second_order(S, ci, kind)
    # rescale S2 rowspace to S's scale so alpha means what it says
    s_scale = S.max() + 1e-8
    Suse = alpha * S + (1.0 - alpha) * S2 * s_scale
    ev = evidence(Suse, ci, fg, low_frac, gamma)
    a = gated_assignment(Suse, ci, ev, gate)
    return masks_from(data, cm, ci, a)


if __name__ == "__main__":
    from harness import load, sweep, save_results
    data = load()
    res = sweep(run, {"alpha": [0.0, 0.3, 0.5, 0.7], "kind": ["pearson", "cosine"]},
                data, tag="profile")
    save_results(res, "/Users/arjunsingla/sam2/variants/results_profile.json")
