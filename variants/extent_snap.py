"""
Extent snapping — fix systematically wrong pick extents (SAM2-only, general).
==============================================================================
Failure mode measured on dice/clover: the clique converges to a SELF-CONSISTENT
wrong extent (dice+tray validates dice+tray in other images; 3-leaf clover
validates 3-leaf clover). Gates that score extent moves against the group's own
picks are circular.

Break: joint coordinate descent over EXTENT with the peaked (top-3) statistic.
For each image, candidates OVERLAPPING the current pick (the object anchor is
already correct) compete by top-3 masked-crop agreement with the other images'
current picks. True object extents are near-duplicates across images (a die is
a die); context unions (die+tray, die+book) differ per image and have no twins.
As images flip to true extents, the signal for the rest sharpens.
"""
import numpy as np
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
import v_reencode as vr
from v_anneal import _qsim
from v_combo import _prep, _cache, weighted_set_sim


def snap(data, picks, rounds=3, iou_min=0.15, pool_cap=24, topk=3, verbose=True):
    U, cr, ci, cm, sp_img, fg = _prep(data, 3)
    key = ("sim", 0.0, 0.1)
    if key not in _cache:
        _cache[key] = weighted_set_sim(U, cr, ci, np.ones(len(U), np.float32))
    G = np.maximum(U @ U.T, 0.0)
    N = len(data)

    # full-res images / label maps for crop encoding
    imgs, lmfs = [], []
    for n in range(N):
        img = np.array(Image.open(data[n]["meta"]["file"]).convert("RGB"))
        lmf = np.array(Image.fromarray(data[n]["label_map"].astype(np.int32), mode="I")
                       .resize((img.shape[1], img.shape[0]), Image.NEAREST))
        imgs.append(img); lmfs.append(lmf)

    dcache = {}

    def desc(n, members):
        keyd = (n, tuple(sorted(members)))
        if keyd not in dcache:
            mfull = np.isin(lmfs[n], list(members))
            dcache[keyd] = (vr._encode_crop(imgs[n], mfull, masked=True)
                            if mfull.any() else np.zeros(256, np.float32))
        return dcache[keyd]

    cur = [sorted(m) for m, _ in picks]

    def top3_agree(n, members):
        v = desc(n, members)
        vals = sorted((max(0.0, float(v @ desc(j, cur[j]))) for j in range(N) if j != n),
                      reverse=True)
        return float(np.mean(vals[:topk]))

    for r in range(rounds):
        changed = False
        for n in range(N):
            lm = data[n]["label_map"]
            pick_mask = np.isin(lm, cur[n])
            # candidate pool: overlap with current pick
            pool = []
            for k in np.where(ci == n)[0]:
                mask = np.isin(lm, list(cm[k]))
                inter = float((mask & pick_mask).sum())
                if inter == 0:
                    continue
                iou = inter / (float((mask | pick_mask).sum()) + 1e-8)
                if iou >= iou_min:
                    pool.append((iou, k))
            if not pool:
                continue
            # cap pool by qsim agreement to bound encodes
            if len(pool) > pool_cap:
                others_rows = [np.concatenate([cr[kk] for kk in np.where(ci == j)[0]
                                               if cm[kk] == frozenset(cur[j])]) if False else None
                               for j in range(N)]
                pool.sort(key=lambda t: -t[0])
                pool = pool[:pool_cap]
            best_mem, best_v = cur[n], top3_agree(n, cur[n])
            for _, k in pool:
                mem = sorted(cm[k])
                if mem == cur[n]:
                    continue
                v = top3_agree(n, mem)
                if v > best_v:
                    best_v, best_mem = v, mem
            if best_mem != cur[n]:
                if verbose:
                    print(f"  round {r} img{n:02d}: {cur[n]} -> {best_mem} (top3 {best_v:.3f})",
                          flush=True)
                cur[n] = best_mem
                changed = True
        if not changed:
            break

    out = []
    for n in range(N):
        lm = data[n]["label_map"]
        out.append((cur[n], np.isin(lm, cur[n])))
    return out


if __name__ == "__main__":
    import importlib
    from harness import load, evaluate
    from v_final import run

    data_path, gt_mod, cache_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    vr.CACHE_DIR = Path(cache_dir); vr.CACHE_DIR.mkdir(exist_ok=True)
    data = load(data_path)
    picks = run(data, K=10, q=0.3, lam=0.15, small_k=5)
    snapped = snap(data, picks)
    gt = importlib.import_module(gt_mod)
    h0, m0, _ = evaluate(picks, data, boxes=gt.BOXES)
    h1, m1, d1 = evaluate(snapped, data, boxes=gt.BOXES)
    print(f"SNAP [{tag}]: before {len(h0)}/{len(data)} miss={m0}", flush=True)
    print(f"SNAP [{tag}]: after  {len(h1)}/{len(data)} miss={m1}", flush=True)
    for d in d1:
        if not d["hit"]:
            print("   ", d, flush=True)
