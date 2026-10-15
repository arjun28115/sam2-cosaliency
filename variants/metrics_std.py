"""
Standard CoSOD evaluation protocol (as in py-sod-metrics / Fan et al. CoSOD toolbox).
Dataset-level: per-image curves over 256 thresholds are averaged over ALL images.
  Fβmax : F from mean precision & mean recall curves, max over thresholds (β²=0.3)
  Eξmax : mean E-measure curve, max over thresholds
  Sα    : mean S-measure of the soft map (α=0.5)
  MAE   : mean absolute error of the soft map
Predictions are soft maps in [0,1]; GT binary.
"""
import numpy as np
from metrics import s_measure

T = 256


def _curves(pred, gt):
    q = np.clip((pred * 255).round(), 0, 255).astype(np.int64)
    hf = np.bincount(q[gt], minlength=T)[::-1].cumsum()[::-1]     # TP(t) = #fg with q>=t
    hb = np.bincount(q[~gt], minlength=T)[::-1].cumsum()[::-1]    # FP(t)
    TP = hf.astype(np.float64); FP = hb.astype(np.float64)
    nG = gt.sum(); N = gt.size
    FN = nG - TP; TN = (N - nG) - FP
    P = np.where(TP + FP > 0, TP / np.maximum(TP + FP, 1), 0.0)
    R = TP / nG if nG > 0 else np.zeros(T)
    # E-measure per threshold, closed form over the 4 (B,G) cells
    if nG == 0:
        E = 1.0 - (TP + FP) / N
    elif nG == N:
        E = (TP + FP) / N
    else:
        mB = (TP + FP) / N; mG = nG / N
        E = np.zeros(T)
        for b, g, cnt in ((1, 1, TP), (1, 0, FP), (0, 1, FN), (0, 0, TN)):
            pb = b - mB; pg = g - mG
            al = 2 * pb * pg / (pb * pb + pg * pg + 1e-12)
            E += cnt * (1 + al) ** 2 / 4
        E /= N
    return P, R, E


def dataset_metrics(preds, gts):
    Ps, Rs, Es, S, M = [], [], [], [], []
    for p, g in zip(preds, gts):
        g = np.asarray(g) > (127 if np.asarray(g).max() > 1 else 0.5)
        p = np.clip(np.asarray(p, np.float64), 0, 1)
        P, R, E = _curves(p, g)
        Ps.append(P); Rs.append(R); Es.append(E)
        S.append(s_measure(p, g)); M.append(np.abs(p - g).mean())
    P = np.mean(Ps, 0); R = np.mean(Rs, 0)
    F = 1.3 * P * R / np.maximum(0.3 * P + R, 1e-12)
    return {"MAE": float(np.mean(M)), "Fmax": float(F.max()),
            "Emax": float(np.mean(Es, 0).max()), "S": float(np.mean(S)), "n": len(preds)}
