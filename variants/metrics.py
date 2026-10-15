"""Standard CoSOD metrics: MAE, F-beta (beta^2=0.3), E-measure, S-measure.
Predictions here are binary masks (valid special case of saliency maps)."""
import numpy as np


def mae(pred, gt):
    return float(np.mean(np.abs(pred.astype(np.float64) - gt.astype(np.float64))))


def fbeta(pred, gt, beta2=0.3):
    p = pred > 0.5; g = gt > 0.5
    tp = float((p & g).sum())
    if p.sum() == 0:
        return 0.0 if g.sum() > 0 else 1.0
    prec = tp / (float(p.sum()) + 1e-12)
    rec = tp / (float(g.sum()) + 1e-12)
    if prec + rec == 0:
        return 0.0
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12)


def s_measure(pred, gt, alpha=0.5):
    """Structure measure (Fan et al. ICCV17): alpha*S_object + (1-alpha)*S_region."""
    pred = pred.astype(np.float64); gt = gt > 0.5
    y = gt.mean()
    if y == 0:
        return 1.0 - pred.mean()
    if y == 1:
        return pred.mean()
    return alpha * _s_object(pred, gt) + (1 - alpha) * _s_region(pred, gt)


def _s_object(pred, gt):
    fg = np.where(gt, pred, 0.0)
    bg = np.where(~gt, 1 - pred, 0.0)
    u = gt.mean()
    return u * _obj(fg, gt) + (1 - u) * _obj(bg, ~gt)


def _obj(pred, region):
    if region.sum() == 0:
        return 0.0
    x = pred[region].mean()
    sx = pred[region].std()
    return 2.0 * x / (x * x + 1.0 + sx + 1e-12)


def _centroid(gt):
    ys, xs = np.where(gt)
    if len(ys) == 0:
        H, W = gt.shape
        return H // 2, W // 2
    return int(round(ys.mean())), int(round(xs.mean()))


def _divide(m, cy, cx):
    return m[:cy, :cx], m[:cy, cx:], m[cy:, :cx], m[cy:, cx:]


def _ssim(p, g):
    g = g.astype(np.float64)
    N = p.size
    if N == 0:
        return 1.0
    x, y = p.mean(), g.mean()
    sx = ((p - x) ** 2).sum() / max(N - 1, 1)
    sy = ((g - y) ** 2).sum() / max(N - 1, 1)
    sxy = ((p - x) * (g - y)).sum() / max(N - 1, 1)
    a = 4 * x * y * sxy
    b = (x * x + y * y) * (sx + sy)
    if a != 0:
        return a / (b + 1e-12)
    return 1.0 if b == 0 else 0.0


def _s_region(pred, gt):
    cy, cx = _centroid(gt)
    H, W = gt.shape
    gs = _divide(gt, cy, cx); ps = _divide(pred, cy, cx)
    ws = [g.size / (H * W) for g in gs]
    return sum(w * _ssim(p, g) for w, p, g in zip(ws, ps, gs))


def e_measure(pred, gt):
    """Enhanced-alignment measure (Fan et al. IJCAI18), binary prediction."""
    p = (pred > 0.5).astype(np.float64); g = (gt > 0.5).astype(np.float64)
    if g.sum() == 0:
        return 1.0 - p.mean()
    if g.sum() == g.size:
        return p.mean()
    dp = p - p.mean(); dg = g - g.mean()
    align = 2 * dp * dg / (dp * dp + dg * dg + 1e-12)
    return float(((align + 1) ** 2 / 4).mean())


def iou(pred, gt):
    """Intersection-over-Union of the binarised prediction and GT."""
    p = pred > 0.5; g = gt > 0.5
    uni = float((p | g).sum())
    return 1.0 if uni == 0 else float((p & g).sum()) / uni


def evaluate_group(preds, gts):
    """Per-group means; preds/gts lists of HxW arrays (pred in [0,1])."""
    out = {"MAE": [], "F": [], "S": [], "E": [], "IoU": []}
    for p, g in zip(preds, gts):
        g = (np.asarray(g) > 127) if np.asarray(g).max() > 1 else (np.asarray(g) > 0.5)
        p = np.asarray(p, np.float64)
        out["MAE"].append(mae(p, g))
        out["F"].append(fbeta(p, g))
        out["S"].append(s_measure(p, g))
        out["E"].append(e_measure(p, g))
        out["IoU"].append(iou(p, g))
    return {k: float(np.mean(v)) for k, v in out.items()}
