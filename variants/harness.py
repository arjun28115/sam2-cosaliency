"""
Shared harness for co-saliency variant experiments  (SAM2-only, general).
=========================================================================
Every variant module must expose:

    def run(data, **params) -> list[(member_club_ids, mask_bool_HxW)]   # one per image

The harness provides:
  * load()                    — the color clock dataset (list of per-image dicts)
  * candidates(data, K)       — multi-granularity union candidates (rows into U,
                                image index, member frozensets) + normalised
                                superpixel embedding matrix U
  * rbd_fg(data, ...)         — RBD boundary-connectivity foreground prior per club
  * evaluate(picks, data)     — strict coverage+purity scoring against GT
  * sweep(run_fn, grid)       — evaluate a param grid, return results sorted by hits

GT lives in gt_color.py (written after visual annotation of the color run).
"""
import itertools
import json
import numpy as np

DATA_PATH = "/Users/arjunsingla/sam2/data/clock_color.npy"


def canonicalize(data):
    """Repair a pipeline artifact: some label_map clubs have no superpixel
    support (created by pixel-level cleanup) — they are invisible to the
    candidate space and break club adjacency. Reassign every orphan pixel to
    the nearest usable club (geodesic via distance transform). General fix,
    no category knowledge."""
    from scipy.ndimage import distance_transform_edt
    for g in data:
        lm = g["label_map"]
        usable = set(int(c) for c in g["club_super_embeddings"].keys()) & \
                 set(int(v) for v in g["sp_to_club"].values())
        orphan = np.isin(lm, [c for c in np.unique(lm) if int(c) not in usable])
        if orphan.any():
            _, (iy, ix) = distance_transform_edt(orphan, return_indices=True)
            lm2 = lm.copy()
            lm2[orphan] = lm[iy[orphan], ix[orphan]]
            g["label_map"] = lm2
    return data


def load(path=DATA_PATH):
    return canonicalize(np.load(path, allow_pickle=True))


# ── candidate construction (multi-granularity connected unions) ───────────────

def _club_adjacency(label_map, club_ids):
    lm = label_map
    adj = {c: set() for c in club_ids}
    for A, B in ((lm[:, :-1], lm[:, 1:]), (lm[:-1, :], lm[1:, :])):
        d = A != B
        for a, b in zip(A[d].ravel(), B[d].ravel()):
            a, b = int(a), int(b)
            if a in adj and b in adj:
                adj[a].add(b); adj[b].add(a)
    return adj


def candidates(data, max_union=3, key="club_super_embeddings"):
    """Returns U (T,256) L2-normalised superpixel embeddings, cand_rows (list of
    row arrays), cand_img (Ncand,), cand_members (list of frozenset club ids)."""
    U, cand_rows, cand_img, cand_members = [], [], [], []
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
        seen = set()

        def add(members):
            fs = frozenset(members)
            if fs in seen:
                return
            seen.add(fs)
            cand_rows.append(np.concatenate([rows_of[c] for c in members]))
            cand_img.append(n); cand_members.append(fs)

        for c in club_ids:
            add([c])
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
    return np.concatenate(U, 0), cand_rows, np.array(cand_img), cand_members


# ── RBD foreground prior ──────────────────────────────────────────────────────

def rbd_fg_clubs(data, key="club_super_embeddings"):
    """{(img, club_id): fg weight} via boundary connectivity."""
    bcs, keys = [], []
    for n, g in enumerate(data):
        lm = g["label_map"]
        frame = np.zeros_like(lm, bool)
        frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
        for cid in g[key].keys():
            m = lm == int(cid)
            area = float(m.sum()) + 1e-6
            bcs.append(float((m & frame).sum()) / np.sqrt(area))
            keys.append((n, int(cid)))
    bcs = np.array(bcs)
    pos = bcs[bcs > 0]
    sigma = np.median(pos) if len(pos) else 1.0
    fg = np.exp(-(bcs ** 2) / (2 * sigma ** 2 + 1e-8))
    return dict(zip(keys, fg.astype(np.float32)))


def rbd_fg_candidates(data, cand_img, cand_members, key="club_super_embeddings"):
    """Area-weighted candidate foreground weight."""
    fgc = rbd_fg_clubs(data, key)
    areas = {}
    for n, g in enumerate(data):
        lm = g["label_map"]
        for cid in g[key].keys():
            areas[(n, int(cid))] = float((lm == int(cid)).sum())
    out = np.empty(len(cand_img), np.float32)
    for k, (n, mem) in enumerate(zip(cand_img, cand_members)):
        w = np.array([areas[(n, c)] for c in mem])
        f = np.array([fgc[(n, c)] for c in mem])
        out[k] = (w * f).sum() / (w.sum() + 1e-8)
    return out


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(picks, data, purity_thresh=0.6, cover_thresh=0.25, verbose=False, boxes=None):
    """Geometric criterion against GT boxes (default gt_color.BOXES):
       purity   : >= purity_thresh of picked mask pixels inside a GT box
       coverage : picked mask covers >= cover_thresh of that box's area
       (both must hold for the SAME box; any box counts in multi-object images)"""
    if boxes is None:
        from gt_color import BOXES
    else:
        BOXES = boxes
    hits, misses, det = [], [], []
    for n, (mem, mask) in enumerate(picks):
        mem = set(int(c) for c in mem)
        H, W = mask.shape
        # union purity: the pick must lie on GT objects (any of them) —
        # fair for multi-instance sets where a pick may span several instances
        um = np.zeros((H, W), bool)
        bms = []
        for (x0, y0, x1, y1) in BOXES[n]:
            bm = np.zeros((H, W), bool)
            bm[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
            um |= bm; bms.append(bm)
        purity = float((mask & um).sum()) / (float(mask.sum()) + 1e-8)
        # coverage: the pick must substantially cover at least ONE instance box
        cover = max(float((mask & bm).sum()) / (float(bm.sum()) + 1e-8) for bm in bms)
        ok = purity >= purity_thresh and cover >= cover_thresh
        (hits if ok else misses).append(n)
        det.append({"img": n, "members": sorted(mem), "purity": round(purity, 3),
                    "cover": round(cover, 3), "hit": bool(ok)})
        if verbose:
            print(f"  {n:02d}: {sorted(mem)} purity={purity:.2f} "
                  f"cover={cover:.2f} {'HIT' if ok else 'MISS'}")
    return hits, misses, det


def sweep(run_fn, grid, data=None, tag=""):
    """grid: dict param -> list of values. Prints and returns results sorted by hits."""
    if data is None:
        data = load()
    keys = list(grid.keys())
    results = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        try:
            picks = run_fn(data, **params)
            hits, misses, det = evaluate(picks, data)
            results.append({"params": params, "hits": len(hits), "misses": misses, "detail": det})
            print(f"[{tag}] {params} -> {len(hits)}/16 miss={misses}", flush=True)
        except Exception as e:
            print(f"[{tag}] {params} -> ERROR {e}", flush=True)
            results.append({"params": params, "hits": -1, "error": str(e)})
    results.sort(key=lambda r: -r["hits"])
    return results


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", path)
