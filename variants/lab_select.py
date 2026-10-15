"""
Selection lab, step 2: try selection rules on cached finalists, score in seconds.
Scoring uses per-club pixel overlap with GT (coarse partition masks, no SAM2
refinement), so absolute numbers are a proxy; relative comparisons are what matter.
"""
import glob
import os
import numpy as np
from PIL import Image

D = "/Users/arjunsingla/sam2/data/coca"


def _rank(M, valid):
    out = np.zeros_like(M, dtype=np.float32)
    v = M[valid]
    if len(v):
        r = np.empty(len(v), np.float32)
        r[np.argsort(v, kind="stable")] = np.arange(1, len(v) + 1) / len(v)
        out[valid] = r
    return out


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (v.max() + 1e-12)


def load_group(tag):
    z = np.load(f"{D}/lab_{tag}.npz", allow_pickle=True)
    data = np.load(f"{D}/{tag}.npy", allow_pickle=True)
    g = {k: z[k] for k in z.files}
    N = int(g["N"])
    inter, carea, gta = [], [], []
    for n in range(N):
        f = str(g["files"][n])
        gt = np.array(Image.open(f.replace("/Images/", "/ground_truth/").rsplit(".", 1)[0] + ".png")
                      .convert("L")) > 127
        lm = data[n]["label_map"]
        if lm.shape != gt.shape:
            lm = np.array(Image.fromarray(lm.astype(np.int32), mode="I")
                          .resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
        nl = int(lm.max()) + 1
        inter.append(np.bincount(lm.ravel(), weights=gt.ravel(), minlength=nl))
        carea.append(np.bincount(lm.ravel(), minlength=nl).astype(np.float64))
        gta.append(float(gt.sum()))
    M = len(g["img"])
    fi = np.zeros(M); fa = np.zeros(M)
    for i in range(M):
        n = g["img"][i]; mem = g["members"][i]
        fi[i] = inter[n][mem].sum(); fa[i] = carea[n][mem].sum()
    g.update(N=N, f_inter=fi, f_area=fa, gt_area=np.array(gta), tag=tag,
             c_inter=inter, c_area=carea)
    tp = f"{D}/trunk_{tag}.npz"
    if os.path.exists(tp):                  # deep SAM2 trunk descriptors (lab_trunk.py)
        t = np.load(tp)
        for k in t.files:
            X = t[k]
            g["S" + k] = np.maximum(X @ X.T, 0.0).astype(np.float32)
    same = g["img"][:, None] == g["img"][None, :]
    g["same"] = same
    ov = np.zeros((M, M), bool)          # finalists sharing a club (same image)
    for i in range(M):
        si = set(g["members"][i])
        for j in np.where(same[i])[0]:
            if j != i and si & set(g["members"][j]):
                ov[i, j] = True
    g["overlap"] = ov
    return g


def all_groups(need_trunk=False):
    tags = sorted(os.path.basename(p)[4:-4] for p in glob.glob(f"{D}/lab_*.npz"))
    if need_trunk:
        tags = [t for t in tags if os.path.exists(f"{D}/trunk_{t}.npz")]
    return [load_group(t) for t in tags]


# ---------- metrics for a single pick per image ----------
def pick_metrics(g, picks):
    ious, fs = [], []
    for n, k in enumerate(picks):
        G = g["gt_area"][n]
        if k < 0:
            ious.append(0.0); fs.append(0.0); continue
        I = g["f_inter"][k]; A = g["f_area"][k]
        u = A + G - I
        ious.append(I / u if u > 0 else 1.0)
        p = I / A if A > 0 else 0.0; r = I / G if G > 0 else 0.0
        fs.append(1.3 * p * r / (0.3 * p + r) if p + r > 0 else 0.0)
    return np.array(ious), np.array(fs)


def oracle_picks(g):
    out = []
    for n in range(g["N"]):
        sel = np.where(g["img"] == n)[0]
        I = g["f_inter"][sel]; A = g["f_area"][sel]; G = g["gt_area"][n]
        out.append(sel[np.argmax(I / np.maximum(A + G - I, 1))])
    return out


# ---------- the pair matrix and the clique search ----------
def pair_matrix(g, fg_pow=0.25, views=("q", "r", "m")):
    valid = ~g["same"]
    P = np.ones(g["Sq"].shape, np.float32)
    for v in views:
        P *= _rank(g["S" + v], valid)
    if fg_pow:
        P *= np.outer(g["fg"], g["fg"]) ** fg_pow
    P[g["same"]] = 0.0
    return P


def clique(g, P, unary, agg="mean", agg_q=0.3, restarts=6, rounds=20):
    """One finalist per image maximising sum_i [ agg_j P(a_i,a_j) + unary(a_i) ]."""
    img = g["img"]; N = g["N"]
    by = [np.where(img == n)[0] for n in range(N)]

    def score_row(k, others):
        v = P[k, others]
        if agg == "mean":
            return v.mean()
        k_ = max(1, int(np.ceil(agg_q * len(v))))
        return np.sort(v)[:k_].mean()              # robust: mean of worst agg_q

    def ascend(a):
        for _ in range(rounds):
            ch = False
            for i in range(N):
                oth = np.array([a[j] for j in range(N) if j != i])
                vals = [score_row(k, oth) + unary[k] for k in by[i]]
                b = by[i][int(np.argmax(vals))]
                if b != a[i]:
                    a[i] = b; ch = True
            if not ch:
                break
        return a

    def J(a):
        return sum(score_row(a[i], np.array([a[j] for j in range(N) if j != i])) + unary[a[i]]
                   for i in range(N))

    inits = []
    a0 = [g["a0"][n] if g["a0"][n] >= 0 else by[n][0] for n in range(N)]
    inits.append(list(a0))
    for r in range(restarts - 1):
        inits.append([by[n][np.argsort(-unary[by[n]])[min(r, len(by[n]) - 1)]] for n in range(N)])
    Pf = P.copy()
    for _ in range(3):                              # pair-seeded inits
        i0, j0 = np.unravel_index(np.argmax(Pf), Pf.shape)
        if Pf[i0, j0] <= 0:
            break
        seed = {img[i0]: i0, img[j0]: j0}
        for n in range(N):
            if n not in seed:
                ss = list(seed.values())
                seed[n] = by[n][int(np.argmax([P[k, ss].mean() for k in by[n]]))]
        inits.append([seed[n] for n in range(N)])
        Pf[i0, j0] = Pf[j0, i0] = 0
    best, bJ = None, -1e18
    for a in inits:
        a = ascend(list(a)); v = J(a)
        if v > bJ:
            bJ, best = v, list(a)
    return best


def run_config(groups, cfg):
    rows = []
    for g in groups:
        P = pair_matrix(g, cfg.get("fg_pow", 0.25), cfg.get("views", ("q", "r", "m")))
        un = cfg.get("lam", 0.15) * _unit(g["ev"])
        if cfg.get("distinct"):
            un = un + cfg["distinct"] * distinctiveness(g, cfg.get("distinct_view", "m"))
        picks = clique(g, P, un, cfg.get("agg", "mean"), cfg.get("agg_q", 0.3))
        iou, f = pick_metrics(g, picks)
        rows.append((g["tag"], iou.mean(), f.mean(), len(iou)))
    return rows


def distinctiveness(g, view="m"):
    """1 - max similarity to a NON-overlapping region of the same image (in [0,1])."""
    S = g["S" + view]
    mask = g["same"] & ~g["overlap"] & ~np.eye(len(S), dtype=bool)
    mx = np.where(mask, S, 0).max(1)
    return 1.0 - mx


def soft_scores(g, P, picks, gamma=2.0):
    """Per-finalist co-saliency score in [0,1]: agreement with the OTHER images'
    picks, normalised by the best finalist in its own image; the pick gets 1."""
    img = g["img"]; N = g["N"]
    s = np.zeros(len(img), np.float32)
    for n in range(N):
        sel = np.where(img == n)[0]
        oth = np.array([picks[j] for j in range(N) if j != n])
        a = P[np.ix_(sel, oth)].mean(1)
        s[sel] = (a / (a.max() + 1e-12)) ** gamma
        s[picks[n]] = 1.0
    return s


def club_curves(g, s, floor=0.0):
    """Std-protocol P/R/E curves per image from a club-level soft map
    (club value = max score of finalists containing it; values < floor -> 0)."""
    T = 256; out = []
    for n in range(g["N"]):
        nl = len(g["c_area"][n]); v = np.zeros(nl)
        for k in np.where(g["img"] == n)[0]:
            if s[k] >= floor:
                m = g["members"][k]; v[m] = np.maximum(v[m], s[k])
        q = np.clip((v * 255).round(), 0, 255).astype(int)
        I = g["c_inter"][n]; A = g["c_area"][n]
        tp = np.bincount(q, weights=I, minlength=T)[::-1].cumsum()[::-1]
        pa = np.bincount(q, weights=A, minlength=T)[::-1].cumsum()[::-1]
        Npx = A.sum(); G = g["gt_area"][n]; fp = pa - tp
        P = np.where(pa > 0, tp / np.maximum(pa, 1), 0.0); R = tp / G if G > 0 else np.zeros(T)
        if 0 < G < Npx:
            mB = pa / Npx; mG = G / Npx; E = np.zeros(T)
            for b, gg, c in ((1, 1, tp), (1, 0, fp), (0, 1, G - tp), (0, 0, Npx - G - fp)):
                pb = b - mB; pg = gg - mG; al = 2 * pb * pg / (pb * pb + pg * pg + 1e-12)
                E += c * (1 + al) ** 2 / 4
            E /= Npx
        else:
            E = 1 - pa / Npx if G == 0 else pa / Npx
        out.append((P, R, E))
    return out


def dataset_std(curves):
    P = np.mean([c[0] for c in curves], 0); R = np.mean([c[1] for c in curves], 0)
    E = np.mean([c[2] for c in curves], 0)
    F = 1.3 * P * R / np.maximum(0.3 * P + R, 1e-12)
    return F.max(), E.max()


def run_std(groups, cfg, soft=True):
    """Selection + std-protocol scoring over ALL images of all groups (club-level proxy)."""
    curves = []
    for g in groups:
        P = pair_matrix(g, cfg.get("fg_pow", 0.25), cfg.get("views", ("q", "r", "m")))
        un = cfg.get("lam", 0.15) * _unit(g["ev"])
        picks = clique(g, P, un, cfg.get("agg", "mean"), cfg.get("agg_q", 0.3))
        if soft:
            s = soft_scores(g, P, picks, cfg.get("gamma", 2.0))
        else:
            s = np.zeros(len(g["img"])); s[picks] = 1.0
        curves += club_curves(g, s, cfg.get("floor", 0.0))
    return dataset_std(curves)


def summarize(rows, name=""):
    iou = np.array([r[1] for r in rows]); f = np.array([r[2] for r in rows])
    w = np.array([r[3] for r in rows])
    print(f"{name:34s} groups={len(rows):2d}  IoU(grp)={iou.mean():.3f}  F(grp)={f.mean():.3f}  "
          f"IoU(img)={np.average(iou, weights=w):.3f}", flush=True)
    return iou.mean()
