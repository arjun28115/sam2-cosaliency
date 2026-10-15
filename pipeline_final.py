"""
Final SAM2 super-embedding pipeline (self-contained)
====================================================
Flow:
  SNIC superpixels on SAM2 embeddings
    -> centroid prompts
    -> SAM2 object proposals
    -> NMS
    -> subtraction into a partition (the prior / object "clubs")
    -> constrain super-embeddings to the prior  (assign, no geometric split)
    -> super-embeddings (256-d per superpixel) + clubbed super-embeddings (256-d per club)

Why it generalizes: object grouping comes from SAM2's own proposals turned into a
partition by SUBTRACTION (smallest object wins each overlapping pixel) — no object-
shape or embedding-similarity thresholds. SNIC provides semantic superpixels; the
prior assigns each superpixel (and its 256-d embedding) to one object.

Output per image (np.save dict, keyed by filename):
  label_map             (H,W) int32     object club id per pixel
  superpixels           (H,W) int32     SNIC superpixel id per pixel (un-split)
  superpixel_ids        (N,)            ids matching the matrix rows below
  super_embeddings      (N,256) float32 one L2-normalised 256-d vector per superpixel
  sp_to_club            {sp_id: club}   which club each superpixel was assigned to
  club_super_embeddings {club:(256,)}   clubbed super-embeddings (size-weighted mean)
  club_embeddings       {club:(256,)}   pixel-based reference (cos ~ 0.98 with above)
  n_clubs, n_superpixels, meta

Run:
  python pipeline_final.py --image_dir /path/to/images --output_dir /path/to/out
"""

import argparse
import os
import warnings
from pathlib import Path

import heapq

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (binary_dilation, binary_fill_holes, binary_opening,
                           center_of_mass, distance_transform_edt, label as cc_label, zoom)
from skimage.morphology import disk
from skimage.segmentation import mark_boundaries

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
SAM2_REPO  = Path(__file__).parent
CHECKPOINT = SAM2_REPO / "checkpoints/sam2.1_hiera_large.pt"
CONFIG     = "configs/sam2.1/sam2.1_hiera_l"

# ── Parameters ──────────────────────────────────────────────────────────────
# Object-dependent (both RELATIVE to image size, so they scale automatically):
MIN_AREA_FRAC    = 0.002    # smallest region kept, as a fraction of image area
OPEN_RADIUS_FRAC = 0.012    # morphological-opening radius, as a fraction of short side
# Superpixels:
BASE_N_SP        = 250      # SNIC superpixels for a 640x426 image; scales with area
BASE_AREA        = 640 * 426
SP_CAP           = 600      # cap superpixel count (4K/8K safe)
SNIC_M           = 10.0     # latent-SNIC compactness: higher = rounder, lower = hugs latent edges
SNIC_TARGET      = 192      # resolution latent SNIC runs at on the 256-d embedding map
# Prompts / proposals:
PROMPT_MODE      = "centroid"  # "centroid" (SNIC centroids) or "grid"
GRID             = 32          # grid size if PROMPT_MODE == "grid"
POINTS_PER_BATCH = 64          # decoder batch size (MPS memory bound)
PROPOSAL_LEVEL   = "part"      # "part" (best-IoU, robust) or "object" (largest mask)
OBJECT_MAX_FRAC  = 0.90        # object-mode: ignore near-total ("whole scene") masks
IOU_THRESH       = 0.85        # NMS threshold
# Compute / cleanup:
WORK_MAXSIDE     = 1280     # run all logic at <= this long-side resolution, then upsample
MASK_ISLAND_FRAC = 0.0005   # drop disconnected speckle islands inside a proposal


# ── Latent SNIC: clustering directly on the dense 256-d SAM2 embedding ────────

def n_superpixels(H, W):
    return int(np.clip(BASE_N_SP * (H * W) / BASE_AREA, 40, SP_CAP))


def snic_latent(feature_map, num_superpixels, m=10.0):
    """
    SNIC on a dense (H,W,C) embedding map. Priority-queue, single pass.
    Distance: D = sqrt( d_latent^2 + (m/S)^2 * d_spatial^2 ), where d_latent is
    Euclidean distance in embedding space and S = sqrt(H*W / K).
    """
    H, W, C = feature_map.shape
    F = feature_map.astype(np.float32)
    S = np.sqrt(H * W / num_superpixels)
    spatial_w2 = (m / S) ** 2
    label_map = np.full((H, W), -1, dtype=np.int32)
    distance_map = np.full((H, W), np.inf, dtype=np.float32)

    g = int(np.sqrt(num_superpixels)); K = g * g
    stats = np.zeros((K, 2 + C + 1), dtype=np.float64)   # sum_x, sum_y, sum_feat(C), count
    xs = np.linspace(S / 2, W - S / 2, g); ys = np.linspace(S / 2, H - S / 2, g)

    queue = []; k = 0
    for yf in ys:
        for xf in xs:
            y = min(max(int(round(yf)), 0), H - 1); x = min(max(int(round(xf)), 0), W - 1)
            stats[k, 0] = x; stats[k, 1] = y; stats[k, 2:2 + C] = F[y, x]; stats[k, -1] = 1.0
            heapq.heappush(queue, (0.0, y, x, k)); k += 1

    NEI = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        dist, y, x, k = heapq.heappop(queue)
        if label_map[y, x] != -1:
            continue
        label_map[y, x] = k
        stats[k, 0] += x; stats[k, 1] += y; stats[k, 2:2 + C] += F[y, x]; stats[k, -1] += 1.0
        cnt = stats[k, -1]; cx = stats[k, 0] / cnt; cy = stats[k, 1] / cnt
        cfeat = stats[k, 2:2 + C] / cnt
        for dy, dx in NEI:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and label_map[ny, nx] == -1:
                diff = F[ny, nx] - cfeat
                D = np.sqrt(float(diff @ diff) + spatial_w2 * ((cx - nx) ** 2 + (cy - ny) ** 2))
                if D < distance_map[ny, nx]:
                    distance_map[ny, nx] = D
                    heapq.heappush(queue, (D, ny, nx, k))

    if (label_map == -1).any():                          # fill rare unlabeled pixels
        _, (iy, ix) = distance_transform_edt(label_map == -1, return_indices=True)
        hole = label_map == -1
        label_map[hole] = label_map[iy[hole], ix[hole]]
    return label_map


def snic_superpixels(predictor, H, W):
    """Latent SNIC directly on the 256-d image_embed (upsampled to SNIC_TARGET)."""
    ie = predictor._features["image_embed"].squeeze(0).permute(1, 2, 0).cpu().float().numpy()  # (64,64,256)
    fmap = zoom(ie, (SNIC_TARGET / ie.shape[0], SNIC_TARGET / ie.shape[1], 1), order=1)
    seg = snic_latent(fmap, n_superpixels(H, W), m=SNIC_M)
    return zoom(seg, (H / SNIC_TARGET, W / SNIC_TARGET), order=0).astype(np.int32)


def superpixel_centroids(segs):
    ids = np.unique(segs)
    coms = center_of_mass(np.ones_like(segs), segs, ids)     # list of (y, x)
    return np.array([[x, y] for (y, x) in coms], dtype=np.float32)


def grid_points(H, W, g):
    ys = np.linspace(0.5, H - 0.5, g); xs = np.linspace(0.5, W - 0.5, g)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


# ── SAM2 object proposals ─────────────────────────────────────────────────────

def clean_mask(m, img_area):
    """Fill holes and drop tiny disconnected islands inside a mask."""
    m = binary_fill_holes(m)
    lab, n = cc_label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel()); keep = np.zeros_like(m)
        for i in range(1, n + 1):
            if sizes[i] >= MASK_ISLAND_FRAC * img_area:
                keep |= (lab == i)
        m = keep
    return m


def _choose_granularity(masks_b, ious_b, img_area):
    if PROPOSAL_LEVEL == "object":
        areas = masks_b.reshape(3, -1).sum(1)
        valid = [j for j in range(3) if areas[j] < OBJECT_MAX_FRAC * img_area]
        return max(valid, key=lambda j: areas[j]) if valid else None
    return int(ious_b.argmax())


def sam_point_proposals(predictor, pts, img_area):
    """(mask, score) proposals for prompt points; image must be set on predictor."""
    proposals = []
    with torch.inference_mode():
        for s in range(0, len(pts), POINTS_PER_BATCH):
            chunk = pts[s:s + POINTS_PER_BATCH]
            coords = torch.as_tensor(chunk, dtype=torch.float32,
                                     device=predictor.device).unsqueeze(1)
            labels = torch.ones((coords.shape[0], 1), dtype=torch.int32, device=predictor.device)
            coords = predictor._transforms.transform_coords(
                coords, normalize=True, orig_hw=predictor._orig_hw[-1])
            masks, ious, _ = predictor._predict(coords, labels, multimask_output=True,
                                                return_logits=False)
            masks = masks.cpu().numpy() > 0.0; ious = ious.cpu().numpy()
            for b in range(masks.shape[0]):
                k = _choose_granularity(masks[b], ious[b], img_area)
                if k is None:
                    continue
                m = clean_mask(masks[b, k], img_area)
                if m.sum() > 0:
                    proposals.append((m, float(ious[b, k])))
    return proposals


def compute_iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u > 0 else 0.0


def nms(proposals, iou_thresh):
    order = sorted(range(len(proposals)), key=lambda i: -proposals[i][1])
    keep, supp = [], set()
    for idx, i in enumerate(order):
        if i in supp:
            continue
        keep.append(proposals[i])
        for j in order[idx + 1:]:
            if j not in supp and compute_iou(proposals[i][0], proposals[j][0]) > iou_thresh:
                supp.add(j)
    return keep


# ── Subtraction into a partition (the prior / clubs) ──────────────────────────

def mask_solidity(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return 0.0
    return m.sum() / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) + 1e-8)


def greedy_objects(proposals, min_area, overlap_tol=0.5):
    masks = [m for m, s in proposals if m.sum() >= min_area]
    masks.sort(key=lambda m: -m.sum())
    kept, covered = [], None
    for m in masks:
        if covered is None:
            kept.append(m); covered = m.copy(); continue
        if (m & covered).sum() / (m.sum() + 1e-8) < overlap_tol:
            kept.append(m); covered |= m
    return kept


def merge_slivers(label, min_area):
    out = label.copy(); changed = True
    while changed:
        changed = False
        for c in np.unique(out):
            m = (out == c)
            if m.sum() >= min_area:
                continue
            nb = np.unique(out[binary_dilation(m) & ~m]); nb = nb[nb != c]
            if len(nb) == 0:
                continue
            out[m] = max(nb, key=lambda n: (out == n).sum()); changed = True
    return out


def aggregate(proposals, H, W):
    """SAM2 proposals -> a proper partition by subtraction (smallest object wins)."""
    img_area = H * W; min_area = MIN_AREA_FRAC * img_area
    objs = greedy_objects(proposals, min_area) if PROPOSAL_LEVEL == "object" \
        else [m for m, s in proposals if m.sum() >= min_area]
    objs.sort(key=lambda m: -m.sum())                 # largest first ...
    label = np.full((H, W), -1, dtype=np.int32)
    for i, m in enumerate(objs):
        label[m] = i                                  # ... smaller painted last -> wins overlaps
    next_id = len(objs)

    unlabeled = (label == -1)                         # pixels no proposal covered
    if unlabeled.any():
        r = max(1, int(OPEN_RADIUS_FRAC * min(H, W)))
        comp, n = cc_label(binary_opening(unlabeled, structure=disk(r)))
        for c in range(1, n + 1):
            blob = (comp == c)
            if blob.sum() >= min_area:                # large background / surface region
                label[blob] = next_id; next_id += 1
    holes = (label == -1)                             # thin boundaries -> nearest region
    if holes.any():
        _, (iy, ix) = distance_transform_edt(holes, return_indices=True)
        label[holes] = label[iy[holes], ix[holes]]

    label = merge_slivers(label, min_area)
    out = np.zeros_like(label)
    for new, old in enumerate(np.unique(label)):
        out[label == old] = new
    return out


# ── Super-embeddings, constrained to the prior by ASSIGNMENT ──────────────────

def _image_embed_upsampled(predictor, H, W):
    f = predictor._features["image_embed"].squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return zoom(f, (H / f.shape[0], W / f.shape[1], 1), order=1)   # (H, W, 256)


def region_embeddings(embed_up, label):
    out = {}
    for c in np.unique(label):
        v = embed_up[label == c].mean(0)
        out[int(c)] = v / (np.linalg.norm(v) + 1e-8)
    return out


def super_embeddings_constrained(predictor, segs, prior):
    """
    Constrain super-embeddings to the prior (assignment, no geometric split):
      - one 256-d super-embedding per RAW SNIC superpixel (mean image_embed, L2-norm)
      - assign each superpixel to its dominant club (majority pixel overlap)
    Returns: ids, matrix (N,256), sizes (N,), sp_to_club {id: club}.
    """
    H, W = segs.shape
    embed_up = _image_embed_upsampled(predictor, H, W)
    ids = np.unique(segs)
    mat = np.zeros((len(ids), 256), dtype=np.float32)
    sizes = np.zeros(len(ids), dtype=np.int64)
    sp_to_club = {}
    for i, sid in enumerate(ids):
        m = (segs == sid)
        v = embed_up[m].mean(0)
        mat[i] = v / (np.linalg.norm(v) + 1e-8)
        sizes[i] = int(m.sum())
        vals, cnts = np.unique(prior[m], return_counts=True)
        sp_to_club[int(sid)] = int(vals[cnts.argmax()])
    return ids, mat, sizes, sp_to_club


def club_from_super(sp_ids, sp_mat, sp_sizes, sp_to_club):
    """Clubbed super-embeddings: size-weighted mean of member superpixels, L2-norm."""
    out = {}
    for club in set(sp_to_club.values()):
        rows = [i for i, sid in enumerate(sp_ids) if sp_to_club[int(sid)] == club]
        w = sp_sizes[rows].astype(np.float32)
        v = (sp_mat[rows] * w[:, None]).sum(0) / (w.sum() + 1e-8)
        out[int(club)] = v / (np.linalg.norm(v) + 1e-8)
    return out


# ── Per-image driver (the full flow) ──────────────────────────────────────────

def process_image(img_full, predictor):
    H0, W0 = img_full.shape[:2]
    scale = WORK_MAXSIDE / max(H0, W0)
    if scale < 1.0:
        work = np.array(Image.fromarray(img_full).resize(
            (max(1, int(W0 * scale)), max(1, int(H0 * scale))), Image.BILINEAR))
    else:
        work = img_full
    H, W = work.shape[:2]; img_area = H * W

    with torch.inference_mode():
        predictor.set_image(work)                          # encode once

    # 1. SNIC superpixels on SAM2 embeddings
    segs = snic_superpixels(predictor, H, W)
    # 2. centroid prompts (or grid)
    pts = superpixel_centroids(segs) if PROMPT_MODE == "centroid" else grid_points(H, W, GRID)
    # 3. SAM2 object proposals  +  4. NMS
    proposals = nms(sam_point_proposals(predictor, pts, img_area), IOU_THRESH)
    # 5. subtraction into a partition (the prior)
    prior = aggregate(proposals, H, W)
    # 6. constrain super-embeddings to the prior (assignment)
    sp_ids, sp_mat, sp_sizes, sp_to_club = super_embeddings_constrained(predictor, segs, prior)
    # 7. clubbed super-embeddings  (+ pixel-based reference)
    club_super = club_from_super(sp_ids, sp_mat, sp_sizes, sp_to_club)
    club_pixel = region_embeddings(_image_embed_upsampled(predictor, H, W), prior)

    if scale < 1.0:
        prior = zoom(prior, (H0 / H, W0 / W), order=0).astype(np.int32)
        segs  = zoom(segs,  (H0 / H, W0 / W), order=0).astype(np.int32)

    return {
        "label_map": prior,
        "superpixels": segs,
        "superpixel_ids": sp_ids,
        "super_embeddings": sp_mat,
        "sp_to_club": sp_to_club,
        "club_super_embeddings": club_super,
        "club_embeddings": club_pixel,
        "n_clubs": int(np.unique(prior).shape[0]),
        "n_superpixels": int(len(sp_ids)),
        "n_proposals": len(proposals),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def colorise(lmap, seed=42):
    rng = np.random.default_rng(seed)
    out = np.zeros((*lmap.shape, 3), dtype=np.uint8)
    for c in np.unique(lmap):
        out[lmap == c] = rng.integers(60, 230, 3)
    return out


def visualise(results, img_paths, title, out_path):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[None, :]
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for i, (res, pth) in enumerate(zip(results, img_paths)):
        img = np.array(Image.open(pth).convert("RGB")); lm = res["label_map"]; col = colorise(lm)
        ov = mark_boundaries((img * 0.55 + col * 0.45).clip(0, 255).astype(np.uint8),
                             lm, color=(1, 1, 1), mode="thick")
        axes[i, 0].imshow(img); axes[i, 0].set_title(pth.name); axes[i, 0].axis("off")
        axes[i, 1].imshow(mark_boundaries(col, lm, color=(1, 1, 1), mode="thick"))
        axes[i, 1].set_title(f"clubs {res['n_clubs']}  ({res['n_superpixels']} superpix)"); axes[i, 1].axis("off")
        axes[i, 2].imshow(ov); axes[i, 2].set_title("overlay"); axes[i, 2].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Final SAM2 super-embedding pipeline")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(SAM2_REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"Loading SAM2 ({Path(args.checkpoint).name}) on {device}...")
    predictor = SAM2ImagePredictor(build_sam2(args.config, args.checkpoint, device=device))

    img_dir = Path(args.image_dir)
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    tag = img_dir.name.replace(" ", "_")
    print(f"Found {len(paths)} images in '{img_dir.name}'\n")
    print(f"{'image':<14}{'clubs':>6}{'superpix':>10}{'cos(club~pix)':>15}")
    print("-" * 45)

    results, store = [], {}
    for i, pth in enumerate(paths):
        img = np.array(Image.open(pth).convert("RGB"))
        res = process_image(img, predictor); results.append(res)
        cs, cb = res["club_super_embeddings"], res["club_embeddings"]
        cos = float(np.mean([cs[c] @ cb[c] for c in cs if c in cb]))
        store[pth.name] = {k: res[k] for k in ("label_map", "superpixels", "superpixel_ids",
                                               "super_embeddings", "sp_to_club",
                                               "club_super_embeddings", "club_embeddings",
                                               "n_clubs", "n_superpixels")}
        store[pth.name]["meta"] = {"file": str(pth), "size": img.shape[:2][::-1]}
        print(f"[{i+1:02d}/{len(paths)}] {pth.name:<8}{res['n_clubs']:>6}{res['n_superpixels']:>10}{cos:>15.4f}")

    np.save(out_dir / f"{tag}_final.npy", store)
    visualise(results, paths, f"Final SAM2 super-embedding pipeline — {img_dir.name}",
              out_dir / f"{tag}_final.png")
    print(f"\nSaved embeddings → {out_dir / (tag + '_final.npy')}")
    print(f"Saved figure     → {out_dir / (tag + '_final.png')}")


if __name__ == "__main__":
    main()
