"""
SPAM-style SAM2 segmentation pipeline (generalizable; super-embeddings output)
------------------------------------------------------------------------------
Instead of re-merging SNIC superpixels with many shape/embedding thresholds,
this trusts SAM2's own object proposals and turns them into a proper partition
by SUBTRACTION (smaller objects sit on top of larger ones), then constrains
SNIC superpixels by that partition. This removes the object-shape-specific
magic numbers of pipeline.py, so it generalizes across object classes.

Method (per image, at working resolution <= WORK_MAXSIDE):
  1. Encode once; build a PCA(20) feature map from SAM2's multi-scale features.
  2. SNIC on that embedding map -> semantic superpixels + centroids.
  3. Prompt SAM2 at each superpixel centroid (PROMPT_MODE="centroid") or a grid.
  4. NMS to dedup near-identical proposals.
  5. SUBTRACTION: paint proposals largest -> smallest so the smallest object
     wins each overlapping pixel -> a partition (the object "clubs").
       - opening separates large background from thin inter-object boundaries
       - remaining thin pixels -> nearest region; slivers absorbed
  6. Constrain superpixels to the clubs (intersection; no parameters).
  7. Super-embeddings: one 256-d SAM2 image_embed per superpixel (L2-normalised).
  8. Club the super-embeddings: size-weighted mean per club -> 256-d per club.

Output per image (np.save dict):
  label_map (H,W), superpixels (H,W), superpixel_ids (N,),
  super_embeddings (N,256), club_super_embeddings {club:(256,)},
  club_embeddings {club:(256,)} (pixel-based; cos ~ 1.0 with club_super).

Only TWO object-dependent parameters, both relative (scale with image size):
  MIN_AREA_FRAC    — smallest region we keep, as a fraction of image area
  OPEN_RADIUS_FRAC — morphological-opening radius, as a fraction of short side

Usage:
    python pipeline_spam.py --image_dir /path/to/images --output_dir /path/to/out
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pysnic.algorithms.snic import snic
from scipy.ndimage import (binary_dilation, binary_fill_holes, binary_opening,
                           center_of_mass, distance_transform_edt, label as cc_label, zoom)
from skimage.morphology import disk
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SAM2_REPO    = Path(__file__).parent
CHECKPOINT   = SAM2_REPO / "checkpoints/sam2.1_hiera_large.pt"
CONFIG       = "configs/sam2.1/sam2.1_hiera_l"

# ── Parameters (only these two are object-dependent, and both are relative) ──
PROMPT_MODE     = "centroid"  # "centroid" = prompt SAM at SNIC superpixel centroids (semantic,
                              #              one prompt per region, fewer SAM calls)
                              # "grid"     = uniform GRID x GRID prompts (paper's blind grid)
GRID            = 32        # uniform GRID x GRID point-prompt grid (paper uses 32)
POINTS_PER_BATCH = 64       # decoder batch size (MPS memory bound)
MIN_AREA_FRAC   = 0.002     # keep proposals / background comps larger than 0.2% of image
OPEN_RADIUS_FRAC = 0.012    # morphological-opening radius as fraction of short side
PROPOSAL_LEVEL  = "part"    # "part"   = best-IoU granularity (robust, always-clean prior) [default]
                            # "object" = whole-object granularity (great on clean multi-object
                            #            scenes; can over-merge interlocked clutter)
OBJECT_MAX_FRAC = 0.90      # ignore a near-total mask (background click) when picking the
                            # object-level granularity — only rejects "the whole scene", general
OVERLAP_TOL     = 0.50      # greedy object selection: skip a proposal if >50% of it is already
                            # covered by a larger kept object (drops sub-parts & duplicates).
                            # About mask nesting, not object shape -> generalizes
# ── Superpixel stage (SPAM: the prior constrains SNIC superpixels) ──
USE_SUPERPIXELS = True      # run SNIC and constrain it by the prior object partition
BASE_N_SP       = 250       # SNIC superpixels for a 640x426 image; scales with area
BASE_AREA       = 640 * 426
SP_CAP          = 600       # cap superpixel count (4K/8K safe)
PCA_DIMS        = 20        # PCA dims of SAM2 features fed to SNIC
SNIC_COMPACT    = 0.005     # near-zero = feature-driven superpixels (semantic, not grid)
SNIC_TARGET     = 256       # resolution SNIC runs at on the feature map
# ── Fixed, object-agnostic ──
IOU_THRESH      = 0.85      # NMS: drop near-duplicate proposals
WORK_MAXSIDE    = 1280      # run everything at this long-side resolution
MASK_ISLAND_FRAC = 0.0005   # drop tiny disconnected islands inside a proposal


def n_superpixels(H, W):
    return int(np.clip(BASE_N_SP * (H * W) / BASE_AREA, 40, SP_CAP))


# ── SAM2 grid proposals ───────────────────────────────────────────────────────

def grid_points(H, W, g):
    ys = np.linspace(0.5, H - 0.5, g)
    xs = np.linspace(0.5, W - 0.5, g)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


def clean_mask(m, img_area):
    m = binary_fill_holes(m)
    lab, n = cc_label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        keep = np.zeros_like(m)
        for i in range(1, n + 1):
            if sizes[i] >= MASK_ISLAND_FRAC * img_area:
                keep |= (lab == i)
        m = keep
    return m


def _choose_granularity(masks_b, ious_b, img_area):
    """Pick which of SAM's 3 masks to keep for one prompt point."""
    if PROPOSAL_LEVEL == "object":
        areas = masks_b.reshape(3, -1).sum(1)
        valid = [j for j in range(3) if areas[j] < OBJECT_MAX_FRAC * img_area]
        if not valid:
            return None
        return max(valid, key=lambda j: areas[j])      # largest = whole object
    return int(ious_b.argmax())                         # best-IoU = fine/part


def sam_point_proposals(predictor, pts, img_area):
    """Return (mask_bool, score) proposals for arbitrary prompt points.
    The image must already be set on the predictor."""
    proposals = []
    with torch.inference_mode():
        for s in range(0, len(pts), POINTS_PER_BATCH):
            chunk = pts[s:s + POINTS_PER_BATCH]
            coords = torch.as_tensor(chunk, dtype=torch.float32,
                                     device=predictor.device).unsqueeze(1)      # (B,1,2)
            labels = torch.ones((coords.shape[0], 1), dtype=torch.int32,
                                device=predictor.device)
            coords = predictor._transforms.transform_coords(
                coords, normalize=True, orig_hw=predictor._orig_hw[-1])
            masks, ious, _ = predictor._predict(
                coords, labels, multimask_output=True, return_logits=False)      # (B,3,H,W)
            masks = masks.cpu().numpy() > 0.0
            ious = ious.cpu().numpy()
            for b in range(masks.shape[0]):
                k = _choose_granularity(masks[b], ious[b], img_area)
                if k is None:
                    continue
                m = clean_mask(masks[b, k], img_area)
                if m.sum() > 0:
                    proposals.append((m, float(ious[b, k])))
    return proposals


def superpixel_centroids(segs):
    ids = np.unique(segs)
    coms = center_of_mass(np.ones_like(segs), segs, ids)       # list of (y, x)
    return np.array([[x, y] for (y, x) in coms], dtype=np.float32)


def compute_iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def nms(proposals, iou_thresh):
    order = sorted(range(len(proposals)), key=lambda i: -proposals[i][1])  # by score
    keep, supp = [], set()
    for idx, i in enumerate(order):
        if i in supp:
            continue
        keep.append(proposals[i])
        for j in order[idx + 1:]:
            if j not in supp and compute_iou(proposals[i][0], proposals[j][0]) > iou_thresh:
                supp.add(j)
    return keep


# ── Aggregation into a proper partition ───────────────────────────────────────

def greedy_objects(proposals, min_area):
    """
    Keep maximal, mostly-disjoint object masks. Process LARGEST-first and add a
    mask only if it is not already >OVERLAP_TOL covered by kept masks — this drops
    sub-parts and duplicates so subtraction does not shatter overlapping proposals
    into slivers, giving object-level regions on clean multi-object scenes.
    Purely about mask coverage — no object-shape assumption.

    Caveat: on scenes where SAM returns one near-full-image "foreground" blob, this
    can over-merge; that object-vs-scene granularity is fundamentally ambiguous
    without supervision. Use PROPOSAL_LEVEL="part" for the robust, always-clean prior.
    """
    masks = [m for m, s in proposals if m.sum() >= min_area]
    masks.sort(key=lambda m: -m.sum())                # largest first
    kept, covered = [], None
    for m in masks:
        if covered is None:
            kept.append(m); covered = m.copy(); continue
        if (m & covered).sum() / (m.sum() + 1e-8) < OVERLAP_TOL:
            kept.append(m); covered |= m
    return kept


def merge_slivers(label, min_area):
    """Absorb any region smaller than min_area into its largest neighbour."""
    out = label.copy()
    changed = True
    while changed:
        changed = False
        for c in np.unique(out):
            m = (out == c)
            if m.sum() >= min_area:
                continue
            border = binary_dilation(m) & ~m
            nb = np.unique(out[border]); nb = nb[nb != c]
            if len(nb) == 0:
                continue
            out[m] = max(nb, key=lambda n: (out == n).sum())
            changed = True
    return out


def aggregate(proposals, H, W):
    img_area = H * W
    min_area = MIN_AREA_FRAC * img_area

    # 1. select object masks
    if PROPOSAL_LEVEL == "object":
        objs = greedy_objects(proposals, min_area)           # maximal, mostly-disjoint
    else:
        objs = [m for m, s in proposals if m.sum() >= min_area]
    # 2. paint largest -> smallest so the smallest object wins overlaps (= subtraction)
    objs.sort(key=lambda m: -m.sum())
    label = np.full((H, W), -1, dtype=np.int32)
    for i, m in enumerate(objs):
        label[m] = i
    next_id = len(objs)

    # 3. unlabeled pixels: opening separates large background from thin boundaries
    unlabeled = (label == -1)
    if unlabeled.any():
        r = max(1, int(OPEN_RADIUS_FRAC * min(H, W)))
        opened = binary_opening(unlabeled, structure=disk(r))
        comp, n = cc_label(opened)
        for c in range(1, n + 1):
            blob = (comp == c)
            if blob.sum() >= min_area:
                label[blob] = next_id            # large background / surface region
                next_id += 1

    # 4. remaining unlabeled (thin boundaries + specks) -> nearest labelled region
    holes = (label == -1)
    if holes.any():
        _, (iy, ix) = distance_transform_edt(holes, return_indices=True)
        label[holes] = label[iy[holes], ix[holes]]

    # 5. absorb leftover slivers from subtraction into their largest neighbour
    label = merge_slivers(label, min_area)

    # relabel consecutively
    out = np.zeros_like(label)
    for new, old in enumerate(np.unique(label)):
        out[label == old] = new
    return out, objs


# ── SNIC superpixels constrained by the prior (full SPAM) ─────────────────────

def pca_feature_map(predictor, H, W):
    """Multi-scale SAM2 features -> PCA(20) feature map at SNIC_TARGET, for SNIC."""
    feats = predictor._features

    def to_t(t):
        a = t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        if a.shape[0] != SNIC_TARGET:
            a = zoom(a, (SNIC_TARGET / a.shape[0], SNIC_TARGET / a.shape[1], 1), order=1)
        return a

    combined = np.concatenate([
        to_t(feats["image_embed"]),
        to_t(feats["high_res_feats"][1]),
        to_t(feats["high_res_feats"][0]),
    ], axis=-1)
    flat = StandardScaler().fit_transform(combined.reshape(-1, combined.shape[-1]))
    pca = PCA(n_components=PCA_DIMS, random_state=0).fit_transform(flat)
    pca -= pca.min(0); pca /= pca.max(0) + 1e-8
    return pca.reshape(SNIC_TARGET, SNIC_TARGET, PCA_DIMS)


def snic_superpixels(predictor, H, W):
    """Semantic SNIC superpixels on the SAM2 embedding space, at image resolution."""
    feat = pca_feature_map(predictor, H, W)
    n_sp = n_superpixels(H, W)
    labels, _, _ = snic(feat.tolist(), n_sp, compactness=SNIC_COMPACT)
    segs = np.array(labels, dtype=np.int32)
    return zoom(segs, (H / SNIC_TARGET, W / SNIC_TARGET), order=0).astype(np.int32)


def constrain_to_prior(segs, prior):
    """
    SPAM constraint: a superpixel may not cross an object boundary. Intersect the
    SNIC superpixels with the prior object partition so every resulting unit lies
    inside exactly one prior object. The prior supplies object boundaries; SNIC
    supplies semantic sub-structure and boundary-respecting detail inside them.
    Returns (fine_superpixels, sp_to_club) where sp_to_club maps each fine unit
    to its prior object (the club it belongs to).
    """
    combo = segs.astype(np.int64) * (int(prior.max()) + 1) + prior.astype(np.int64)
    fine = np.zeros_like(segs)
    sp_to_club = {}
    for new, old in enumerate(np.unique(combo)):
        m = (combo == old)
        fine[m] = new
        sp_to_club[new] = int(prior[m][0])     # the object this fine unit sits in
    return fine, sp_to_club


# ── Embeddings ────────────────────────────────────────────────────────────────

def _image_embed_upsampled(predictor, H, W):
    feats = predictor._features["image_embed"].squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return zoom(feats, (H / feats.shape[0], W / feats.shape[1], 1), order=1)   # (H, W, 256)


def region_embeddings(embed_up, label):
    """Mean SAM2 image_embed per region, L2-normalised -> {id: (256,)}."""
    out = {}
    for c in np.unique(label):
        v = embed_up[label == c].mean(0)
        out[int(c)] = v / (np.linalg.norm(v) + 1e-8)
    return out


def club_embeddings(predictor, label):
    H, W = label.shape
    return region_embeddings(_image_embed_upsampled(predictor, H, W), label)


def super_embeddings(predictor, segs):
    """
    'Super-embeddings' — the embedding analogue of superpixels. Each superpixel
    gets one 256-d SAM2 image_embed vector (mean over its pixels, L2-normalised).
    Returns (ids, matrix, sizes) where matrix[i] is the 256-d embedding of
    superpixel ids[i] and sizes[i] is its pixel count.
    """
    H, W = segs.shape
    embed_up = _image_embed_upsampled(predictor, H, W)
    ids = np.unique(segs)
    mat = np.zeros((len(ids), 256), dtype=np.float32)
    sizes = np.zeros(len(ids), dtype=np.int64)
    for i, sid in enumerate(ids):
        m = (segs == sid)
        v = embed_up[m].mean(0)
        mat[i] = v / (np.linalg.norm(v) + 1e-8)
        sizes[i] = int(m.sum())
    return ids, mat, sizes


def club_from_super(sp_ids, sp_mat, sp_sizes, sp_to_club):
    """
    Club the super-embeddings: aggregate each club's member superpixel embeddings
    into one 256-d club vector (pixel-size-weighted mean, L2-normalised). This is
    the two-level hierarchy pixel -> superpixel -> club, built bottom-up from the
    super-embeddings rather than re-reading pixels.
    """
    out = {}
    for club in set(sp_to_club.values()):
        rows = [i for i, sid in enumerate(sp_ids) if sp_to_club[int(sid)] == club]
        w = sp_sizes[rows].astype(np.float32)
        v = (sp_mat[rows] * w[:, None]).sum(0) / (w.sum() + 1e-8)
        out[int(club)] = v / (np.linalg.norm(v) + 1e-8)
    return out


# ── Per-image driver ──────────────────────────────────────────────────────────

def process_image(img_full, predictor):
    H0, W0 = img_full.shape[:2]
    scale = WORK_MAXSIDE / max(H0, W0)
    if scale < 1.0:
        work = np.array(Image.fromarray(img_full).resize(
            (max(1, int(W0 * scale)), max(1, int(H0 * scale))), Image.BILINEAR))
    else:
        work = img_full
    H, W = work.shape[:2]

    img_area = H * W
    with torch.inference_mode():
        predictor.set_image(work)                          # encode once; features reused below

    # 1. SNIC superpixels on the SAM2 embedding space (needed now if prompting at centroids)
    segs = snic_superpixels(predictor, H, W) if (USE_SUPERPIXELS or PROMPT_MODE == "centroid") else None

    # 2. SAM2 object proposals — prompt at superpixel centroids or a uniform grid
    if PROMPT_MODE == "centroid":
        pts = superpixel_centroids(segs)                   # one semantic prompt per superpixel
    else:
        pts = grid_points(H, W, GRID)
    proposals = sam_point_proposals(predictor, pts, img_area)
    proposals = nms(proposals, IOU_THRESH)

    # 3. subtraction prior -> object partition (the "clubs")
    prior, objs = aggregate(proposals, H, W)
    embs = club_embeddings(predictor, prior)        # pixel-based club embedding

    # 4. constrain superpixels by the prior (never cross an object boundary)
    superpixels = None
    sp_ids = sp_mat = None
    club_super = embs
    if USE_SUPERPIXELS and segs is not None:
        superpixels, sp_to_club = constrain_to_prior(segs, prior)
        # 4b. super-embeddings: one 256-d SAM2 vector per (constrained) superpixel
        sp_ids, sp_mat, sp_sizes = super_embeddings(predictor, superpixels)
        # 4c. club the super-embeddings: pixel -> superpixel -> club
        club_super = club_from_super(sp_ids, sp_mat, sp_sizes, sp_to_club)

    if scale < 1.0:
        prior = zoom(prior, (H0 / H, W0 / W), order=0).astype(np.int32)
        if superpixels is not None:
            superpixels = zoom(superpixels, (H0 / H, W0 / W), order=0).astype(np.int32)

    return {
        "label_map": prior,                 # object-level clubs
        "superpixels": superpixels,         # fine, boundary-respecting superpixels
        "club_embeddings": embs,            # 256-d per club (pixel-based)
        "club_super_embeddings": club_super,  # 256-d per club, clubbed from super-embeddings
        "superpixel_ids": sp_ids,           # superpixel id for each row of the matrix below
        "super_embeddings": sp_mat,         # (N_superpixels, 256) — one 256-d vector per superpixel
        "n_clubs": int(np.unique(prior).shape[0]),
        "n_superpixels": int(np.unique(superpixels).shape[0]) if superpixels is not None else 0,
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
        img = np.array(Image.open(pth).convert("RGB"))
        lm = res["label_map"]
        col = colorise(lm)
        overlay = mark_boundaries((img * 0.55 + col * 0.45).clip(0, 255).astype(np.uint8),
                                  lm, color=(1, 1, 1), mode="thick")
        axes[i, 0].imshow(img); axes[i, 0].set_title(pth.name); axes[i, 0].axis("off")
        axes[i, 1].imshow(mark_boundaries(col, lm, color=(1, 1, 1), mode="thick"))
        axes[i, 1].set_title(f"Regions ({res['n_clubs']})  [{res['n_proposals']} proposals]")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay); axes[i, 2].set_title("Overlay"); axes[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SPAM-style object-proposal aggregation pipeline")
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

    results, store = [], {}
    print(f"{'image':<14}{'clubs':>6}{'superpix':>10}{'proposals':>11}")
    print("-" * 41)
    for i, pth in enumerate(paths):
        img = np.array(Image.open(pth).convert("RGB"))
        res = process_image(img, predictor)
        results.append(res)
        # full three-level output: pixel -> super-embedding -> clubbed super-embedding
        store[pth.name] = {
            "label_map":             res["label_map"],              # (H,W) object clubs
            "superpixels":           res["superpixels"],            # (H,W) fine units
            "superpixel_ids":        res["superpixel_ids"],         # (N,)
            "super_embeddings":      res["super_embeddings"],       # (N,256) per superpixel
            "club_super_embeddings": res["club_super_embeddings"],  # {club:(256,)} clubbed from super
            "club_embeddings":       res["club_embeddings"],        # {club:(256,)} pixel-based
            "n_clubs":               res["n_clubs"],
            "n_superpixels":         res["n_superpixels"],
            "meta": {"file": str(pth), "size": img.shape[:2][::-1]},
        }
        print(f"[{i+1:02d}/{len(paths)}] {pth.name:<8}{res['n_clubs']:>6}{res['n_superpixels']:>10}{res['n_proposals']:>11}")

    np.save(out_dir / f"{tag}_spam.npy", store)
    visualise(results, paths, f"SPAM-style aggregation — {img_dir.name}", out_dir / f"{tag}_spam.png")
    print(f"\nSaved embeddings → {out_dir / (tag + '_spam.npy')}")
    print(f"  per image: label_map, superpixels, super_embeddings (N,256),")
    print(f"             club_super_embeddings + club_embeddings (256-d each)")
    print(f"Saved figure     → {out_dir / (tag + '_spam.png')}")


if __name__ == "__main__":
    main()
