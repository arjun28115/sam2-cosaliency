"""
Latent SNIC — SNIC clustering directly on a dense D-dim SAM2 embedding map.
Priority-queue, single-pass region growing. Distance combines latent (semantic)
and spatial terms:  D = sqrt( d_latent^2 + (m/S)^2 * d_spatial^2 ).
"""
import heapq
import numpy as np


def snic_latent(feature_map, num_superpixels, m=10.0):
    """
    feature_map : (H, W, C) dense embedding (C arbitrary, e.g. 256).
    num_superpixels : target K (grid is floor(sqrt(K))^2 seeds).
    m : compactness (higher -> rounder; lower -> hugs latent boundaries).
    Returns label_map (H, W) int32.
    """
    H, W, C = feature_map.shape
    F = feature_map.astype(np.float32)
    N = H * W
    S = np.sqrt(N / num_superpixels)
    spatial_w2 = (m / S) ** 2

    label_map = np.full((H, W), -1, dtype=np.int32)
    distance_map = np.full((H, W), np.inf, dtype=np.float32)

    # per-cluster running sums: [sum_x, sum_y, sum_feat(C), count]
    g = int(np.sqrt(num_superpixels))
    K = g * g
    stats = np.zeros((K, 2 + C + 1), dtype=np.float64)

    xs = np.linspace(S / 2, W - S / 2, g)
    ys = np.linspace(S / 2, H - S / 2, g)

    queue = []
    k = 0
    for yf in ys:
        for xf in xs:
            y, x = int(round(yf)), int(round(xf))
            y = min(max(y, 0), H - 1); x = min(max(x, 0), W - 1)
            stats[k, 0] = x; stats[k, 1] = y
            stats[k, 2:2 + C] = F[y, x]; stats[k, -1] = 1.0
            heapq.heappush(queue, (0.0, y, x, k))   # seed itself, distance 0
            k += 1

    NEI = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        dist, y, x, k = heapq.heappop(queue)
        if label_map[y, x] != -1:
            continue
        label_map[y, x] = k

        # online centroid update
        stats[k, 0] += x; stats[k, 1] += y
        stats[k, 2:2 + C] += F[y, x]; stats[k, -1] += 1.0
        cnt = stats[k, -1]
        cx = stats[k, 0] / cnt; cy = stats[k, 1] / cnt
        cfeat = stats[k, 2:2 + C] / cnt

        for dy, dx in NEI:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and label_map[ny, nx] == -1:
                diff = F[ny, nx] - cfeat
                d_lat2 = float(diff @ diff)
                d_sp2 = (cx - nx) ** 2 + (cy - ny) ** 2
                D = np.sqrt(d_lat2 + spatial_w2 * d_sp2)
                if D < distance_map[ny, nx]:
                    distance_map[ny, nx] = D
                    heapq.heappush(queue, (D, ny, nx, k))

    # fill any unlabeled (rare) with nearest labeled
    if (label_map == -1).any():
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(label_map == -1, return_indices=True)
        label_map[label_map == -1] = label_map[iy[label_map == -1], ix[label_map == -1]]
    return label_map
