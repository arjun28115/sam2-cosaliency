"""
SAM2 Superpixel Clubbing Pipeline
----------------------------------
Pipeline:
  1. Extract multi-scale SAM2 features (image_embed + high_res_feats)
  2. PCA reduce + SNIC on embedding space → semantic superpixels
  3. SAM2 centroid prompts → object masks (SP-relative bleed filter, smallest-valid selection)
  4. NMS to deduplicate masks
  5. Assign superpixels to clubs via mask overlap
  6. Unassigned SPs → nearest club by embedding cosine similarity
  7. Same-parent merge → collapse fragments of same object
  8. Small-club merge → absorb tiny regions
  9. Edge-aware embedding merge → fuse similar adjacent clubs
 10. Export clubbed SAM2 embeddings (256-d, L2-normalised) per club

Usage:
    python pipeline.py --image_dir /path/to/images --output_dir /path/to/output

Output:
    <output_dir>/
        <category>_final_all16.png    — grid overview of all images
        <category>_final_detail4.png  — detailed 5-col view of first 4 images
        <category>_final.npy          — embeddings dict keyed by filename
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pysnic.algorithms.snic import snic
from scipy.ndimage import (binary_dilation, binary_fill_holes, center_of_mass,
                           label as cc_label, zoom)
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────
SAM2_REPO      = Path(__file__).parent          # assumes script lives in sam2/ repo
CHECKPOINT     = SAM2_REPO / "checkpoints/sam2.1_hiera_large.pt"
CONFIG         = "configs/sam2.1/sam2.1_hiera_l"
BASE_N_SP      = 150        # superpixels for a 640×426 image; scales with area
BASE_AREA      = 640 * 426
SP_CAP         = 400        # hard cap on superpixels → bounds SAM2 calls + NMS at 4K/8K
                            # (SAM2 resizes internally, so density past this adds little)
PCA_DIMS       = 20
SNIC_COMPACT   = 0.005      # near-zero = feature-driven, not grid-driven
IOU_THRESH     = 0.75       # NMS threshold
MAX_SP_RATIO   = 10.0       # reject masks >10x the area of the prompting superpixel (bleed)
MIN_MASK_FRAC  = 0.005      # reject masks covering <0.5% of image (noise)
GROUP_MAX_FRAC = 0.28       # object-grouping masks: allow up to 28% of image (reject scene-bleed)
CONTAIN_THR    = 0.60       # club is "in" a grouping mask if >=60% of it lies inside
GROUP_FILL_THR = 0.80       # grouping mask must be >=80% filled by its contained clubs (true object)
GROUP_CC_THR   = 0.90       # grouping mask must be a single blob: largest connected comp >=90%
GROUP_SOLIDITY = 0.34       # grouping mask must fill >=34% of its bbox: keeps irregular single
                            # objects (clock + bells/handle ~0.35) but rejects sprawling masks
                            # that span two separate objects + the gap between them (~0.22-0.29)
CONTAIN_SIM_THR = 0.65      # containment merge requires embeddings this similar: one object's
                            # parts (~0.80) merge, but a different object touching it (~0.55-0.59) does not
SNAP_SOLIDITY  = 0.55       # mask-snap: only clean, tight object masks (solidity >= this) force
                            # their interior into a single club — fixes straddling clubs in clutter
SNAP_SIM_THR   = 0.65       # mask-snap only absorbs clubs whose embedding is this similar to the
                            # mask's dominant club — keeps one object's parts together (~0.80) but
                            # refuses to swallow a different object touching it (e.g. toy vs clock ~0.55)
MIN_CLUB_PX    = 2500       # absorb clubs smaller than this into neighbours
EMBED_SIM_THR  = 0.95       # cosine sim threshold for embedding merge (conservative — only
                            # near-identical regions; object grouping is handled by Stage 7b)
EDGE_THR       = 0.12       # Sobel edge threshold for boundary-aware merge
TARGET         = 256        # resolution for SNIC feature map
WORK_MAXSIDE   = 1280       # run all clubbing/merge logic at this long-side resolution, then
                            # upsample the final label map — keeps full-res arrays out of the
                            # Python merge loops (the real 4K bottleneck). SAM2 resizes to 1024px
                            # internally anyway, so there is no quality loss from this.
MASK_ISLAND_FRAC = 0.001    # mask cleanup: drop disconnected islands smaller than this frac of image


# ── Helpers ───────────────────────────────────────────────────────────────────

def n_superpixels(H, W):
    # Scale with area but cap so SAM2 prompt calls and NMS stay bounded on 4K/8K inputs.
    return int(np.clip(BASE_N_SP * (H * W) / BASE_AREA, 40, SP_CAP))


def compute_iou(m1, m2):
    i = (m1 & m2).sum()
    u = (m1 | m2).sum()
    return float(i) / float(u) if u > 0 else 0.0


def nms_masks(masks, scores, iou_thresh):
    order = np.argsort(scores)[::-1]
    keep, supp = [], set()
    for i in order:
        if i in supp:
            continue
        keep.append(i)
        for j in order:
            if j != i and j not in supp:
                if compute_iou(masks[i], masks[j]) > iou_thresh:
                    supp.add(j)
    return keep


def select_mask(masks, scores, cx, cy, img_area, sp_area):
    """
    Pick the smallest SAM2 mask that:
      - Contains the centroid point (cx, cy)
      - Is not a bleed mask (>MAX_SP_RATIO times the prompting superpixel area)
      - Covers at least MIN_MASK_FRAC of the image (noise filter)
    Falls back to highest-score valid mask if no mask contains the point.
    """
    cy_i = int(np.clip(round(cy), 0, masks[0].shape[0] - 1))
    cx_i = int(np.clip(round(cx), 0, masks[0].shape[1] - 1))
    valid = []
    for i, m in enumerate(masks):
        mask_area = m.sum()
        if mask_area / img_area < MIN_MASK_FRAC:          # too small — noise
            continue
        if mask_area / (sp_area + 1e-8) > MAX_SP_RATIO:  # bleed — mask >> SP
            continue
        valid.append((i, int(mask_area), float(scores[i]), bool(m[cy_i, cx_i])))
    if not valid:
        return None, None
    with_pt = [(i, a, s) for i, a, s, has in valid if has]
    if with_pt:
        return min(with_pt, key=lambda x: x[1])[:2]
    best = max(valid, key=lambda x: x[2])
    return best[0], best[2]


def clean_mask(m, img_area):
    """
    Post-process a raw SAM2 mask: fill interior holes and drop tiny disconnected
    speckle islands. Turns ragged 'swiss-cheese' masks (common on textured/low-
    contrast regions) into clean single blobs. Per-mask cost is independent of how
    the rest of the pipeline scales, so it is safe at 4K/8K.
    """
    m = binary_fill_holes(m)
    lab, n = cc_label(m)
    if n > 1:
        min_px = MASK_ISLAND_FRAC * img_area
        sizes = np.bincount(lab.ravel())
        keep = np.zeros_like(m)
        for i in range(1, n + 1):
            if sizes[i] >= min_px:
                keep |= (lab == i)
        m = keep
    return m


def select_group_mask(masks, scores, cx, cy, img_area):
    """
    Pick the LARGEST mask that contains the centroid and stays under
    GROUP_MAX_FRAC of the image. These object-level masks (e.g. a whole
    clock) are used only to decide which fine clubs belong to one object;
    they are deliberately allowed to be much larger than the prompting SP.
    """
    cy_i = int(np.clip(round(cy), 0, masks[0].shape[0] - 1))
    cx_i = int(np.clip(round(cx), 0, masks[0].shape[1] - 1))
    cand = []
    for i, m in enumerate(masks):
        area = m.sum()
        frac = area / img_area
        if frac > GROUP_MAX_FRAC or frac < MIN_MASK_FRAC:
            continue
        if not bool(m[cy_i, cx_i]):
            continue
        # Single-object check: mask must be one connected blob, not two separate
        # objects (e.g. two clocks) that SAM2 lumped into one prediction.
        lab, n = cc_label(m)
        if n > 1:
            largest = max(np.bincount(lab.ravel())[1:])
            if largest / (area + 1e-8) < GROUP_CC_THR:
                continue
        # Solidity check: a compact single object fills most of its bounding box;
        # a mask spanning two objects + the gap between them does not.
        ys, xs = np.where(m)
        bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        if area / (bbox_area + 1e-8) < GROUP_SOLIDITY:
            continue
        cand.append((i, int(area)))
    if not cand:
        return None
    return max(cand, key=lambda x: x[1])[0]   # largest valid object-level mask


def containment_merge(club_label, group_masks, embed_up):
    """
    Merge the clubs substantially contained (>=CONTAIN_THR) within the same SAM2
    object-level mask into that mask's DOMINANT (largest) club. The object mask
    lets us merge across internal edges (clock rim) and material changes (white
    face vs metallic body).

    Crucially, every absorbed club is gated by similarity to the DOMINANT club —
    not pairwise to its neighbour. Pairwise chaining could bridge two distinct
    objects through an intermediate club (e.g. a toy's arm wrapped over a clock),
    fusing them. Snapping to the dominant club instead means a club unlike the
    main object (toy ~0.55 vs clock) is never pulled in, while the object's own
    parts (~0.80) still merge.
    """
    out = club_label.copy()

    def club_emb(c):
        v = embed_up[out == c].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    changed = True
    while changed:
        changed = False
        for gm in group_masks:
            gm_area = gm.sum()
            inside = [c for c in np.unique(out)
                      if ((out == c) & gm).sum() / ((out == c).sum() + 1e-8) >= CONTAIN_THR]
            if len(inside) < 2:
                continue
            # Guard: contained clubs must fill most of the mask (true object, not loose scene mask)
            covered = sum((out == c).sum() for c in inside)
            if covered / (gm_area + 1e-8) < GROUP_FILL_THR:
                continue
            # dominant (largest) contained club = the object's main body
            dom = max(inside, key=lambda c: (out == c).sum())
            dom_emb = club_emb(dom)
            # absorb the other contained clubs into the dominant one, gated by
            # similarity to the dominant club (breaks transitive object-bridging)
            for c in inside:
                if c == dom:
                    continue
                if float(club_emb(c) @ dom_emb) < CONTAIN_SIM_THR:
                    continue
                out[out == c] = dom
                changed = True
            if changed:
                break
    return out


def mask_solidity(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return 0.0
    bbox = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    return m.sum() / (bbox + 1e-8)


def mask_snap(club_label, group_masks, min_solidity, embed_up):
    """
    Edge-snapping fix for cluttered scenes. For each CLEAN, TIGHT object mask
    (solidity >= min_solidity), force the part of the image inside the mask that
    belongs to clubs mostly-inside it into ONE club. This fixes the case where
    SNIC superpixels straddle an object's edge in low-contrast clutter (e.g. a
    small clock crammed against a bag), which otherwise blocks the containment
    merge. The high solidity bar means only single, compact objects trigger a
    snap — sprawling masks that span two objects (low solidity) are ignored.

    A similarity gate (SNAP_SIM_THR) protects the case where a different object
    is *touching* the target (zero gap, so solidity stays high): only clubs whose
    embedding is similar to the mask's dominant club are absorbed, so e.g. a toy
    figure hugging a clock is not swallowed into the clock.

    Operates on whole-image masks (not per-superpixel loops), so its cost scales
    with the number of grouping masks, not image resolution — safe for 4K/8K.
    """
    out = club_label.copy()

    def club_emb(c):
        v = embed_up[out == c].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    tight = [g for g in group_masks if mask_solidity(g) >= min_solidity]
    tight.sort(key=lambda g: g.sum())          # small, tight objects first
    for gm in tight:
        # clubs that are themselves majority-inside this mask (don't pull in spill-over)
        inside = []
        for c in np.unique(out):
            club = (out == c)
            if (club & gm).sum() / (club.sum() + 1e-8) >= 0.5:
                inside.append(c)
        if len(inside) < 2:
            continue
        # dominant club inside the mask = the object's main body
        areas = {c: (out == c).sum() for c in inside}
        dom = max(areas, key=areas.get)
        dom_emb = club_emb(dom)
        # only absorb clubs whose embedding matches the dominant one (same object)
        members = [c for c in inside if float(club_emb(c) @ dom_emb) >= SNAP_SIM_THR]
        if len(members) < 2:
            continue
        region = np.isin(out, members) & gm
        if region.sum() == 0:
            continue
        out[region] = dom
    return out


def merge_small_clubs(label_map, min_px):
    out = label_map.copy()
    changed = True
    while changed:
        changed = False
        for c in np.unique(out):
            m = (out == c)
            if m.sum() >= min_px:
                continue
            border = binary_dilation(m) & ~m
            nb = np.unique(out[border])
            nb = nb[nb != c]
            if len(nb) == 0:
                continue
            out[m] = max(nb, key=lambda n: (out == n).sum())
            changed = True
    return out


def same_parent_merge(club_label, sp_parent, sp_ids, segs_full):
    """
    Merge adjacent clubs whose superpixels share the same original SAM2 mask.
    Fixes fragmentation of objects with fine interior detail (e.g. clock face).
    """
    club_dominant_parent = {}
    for c in np.unique(club_label):
        members = [s for s in sp_ids if np.any(club_label[segs_full == s] == c)]
        parents = [sp_parent.get(s, -1) for s in members]
        parents = [p for p in parents if p >= 0]
        if parents:
            club_dominant_parent[c] = max(set(parents), key=parents.count)

    out = club_label.copy()
    changed = True
    while changed:
        changed = False
        clubs = np.unique(out)
        for i in range(len(clubs)):
            for j in range(i + 1, len(clubs)):
                ci, cj = clubs[i], clubs[j]
                pi = club_dominant_parent.get(ci, -1)
                pj = club_dominant_parent.get(cj, -1)
                if pi < 0 or pj < 0 or pi != pj:
                    continue
                mask_i = (out == ci)
                if not (binary_dilation(mask_i) & (out == cj)).any():
                    continue
                keep, drop = (ci, cj) if mask_i.sum() >= (out == cj).sum() else (cj, ci)
                out[out == drop] = keep
                club_dominant_parent[keep] = pi
                club_dominant_parent.pop(drop, None)
                changed = True
                break
            if changed:
                break
    return out


def edge_aware_embed_merge(club_label, club_embs, edge_map, sim_thr, edge_thr):
    """
    Merge adjacent clubs if:
      - Their embeddings have cosine similarity > sim_thr
      - No strong image edge (Sobel > edge_thr) lies at their shared boundary
    """
    clubs = list(club_embs.keys())
    mapping = {c: c for c in clubs}

    def root(c):
        while mapping[c] != c:
            c = mapping[c]
        return c

    for i in range(len(clubs)):
        for j in range(i + 1, len(clubs)):
            ri, rj = root(clubs[i]), root(clubs[j])
            if ri == rj:
                continue
            if float(club_embs[ri] @ club_embs[rj]) <= sim_thr:
                continue
            mask_i = (club_label == ri)
            border = binary_dilation(mask_i) & (club_label == rj)
            if border.sum() == 0:
                continue
            if float(edge_map[border].mean()) > edge_thr:
                continue
            keep, drop = (ri, rj) if mask_i.sum() >= (club_label == rj).sum() else (rj, ri)
            for c in clubs:
                if root(c) == drop:
                    mapping[c] = keep

    out = club_label.copy()
    for c in clubs:
        r = root(c)
        if r != c:
            out[club_label == c] = r
    return out


def to_target(t, target=TARGET):
    arr = t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    if arr.shape[0] != target:
        arr = zoom(arr, (target / arr.shape[0], target / arr.shape[1], 1), order=1)
    return arr


def extract_features(predictor, img_np):
    """
    Returns:
      feat_map   (TARGET, TARGET, PCA_DIMS) — normalised, used for SNIC
      pca_full   (TARGET, TARGET, PCA_DIMS) — raw PCA, used for SP embeddings
      embed_up   (H, W, 256)               — image_embed at original resolution
    """
    H, W = img_np.shape[:2]
    with torch.inference_mode():
        predictor.set_image(img_np)
        feats = predictor._features
        combined = np.concatenate([
            to_target(feats["image_embed"]),
            to_target(feats["high_res_feats"][1]),
            to_target(feats["high_res_feats"][0]),
        ], axis=-1)
        raw_embed = feats["image_embed"].squeeze(0).permute(1, 2, 0).cpu().float().numpy()

    flat    = StandardScaler().fit_transform(combined.reshape(-1, combined.shape[-1]))
    pca_out = PCA(n_components=PCA_DIMS, random_state=0).fit_transform(flat)

    pca_norm = pca_out.copy()
    pca_norm -= pca_norm.min(0)
    pca_norm /= pca_norm.max(0) + 1e-8

    embed_up = zoom(raw_embed, (H / raw_embed.shape[0], W / raw_embed.shape[1], 1), order=1)
    return pca_norm.reshape(TARGET, TARGET, PCA_DIMS), pca_out.reshape(TARGET, TARGET, PCA_DIMS), embed_up


def compute_club_embeddings(embed_up, club_label):
    """Mean SAM2 image_embed per club, L2-normalised → (256,) per club."""
    result = {}
    for c in np.unique(club_label):
        v = embed_up[club_label == c].mean(0)
        result[int(c)] = v / (np.linalg.norm(v) + 1e-8)
    return result


def relabel(club_label):
    tmp = club_label.copy()
    for new_id, old_id in enumerate(np.unique(tmp)):
        club_label[tmp == old_id] = new_id
    return club_label


def colorise(lmap, seed=42):
    rng = np.random.default_rng(seed)
    out = np.zeros((*lmap.shape, 3), dtype=np.uint8)
    for c in np.unique(lmap):
        out[lmap == c] = rng.integers(60, 230, 3)
    return out


# ── Core pipeline for a single image ─────────────────────────────────────────

def process_image(img_np, predictor):
    """
    Run the full pipeline at a bounded working resolution (WORK_MAXSIDE) and
    upsample the label map back to the original size. All the merge/assignment
    stages do full-array boolean ops in Python loops, so running them on a raw
    4K/8K image is the real bottleneck — downscaling first keeps them fast with
    no quality loss (SAM2 resizes to 1024px internally regardless).
    """
    H0, W0 = img_np.shape[:2]
    scale = WORK_MAXSIDE / max(H0, W0)
    if scale < 1.0:
        work = np.array(Image.fromarray(img_np).resize(
            (max(1, int(W0 * scale)), max(1, int(H0 * scale))), Image.BILINEAR))
    else:
        work = img_np

    res = _process_image_core(work, predictor)

    if scale < 1.0:
        # upsample label map + SNIC map to original resolution (nearest = no label blending)
        lm = res["label_map"]
        res["label_map"] = zoom(lm, (H0 / lm.shape[0], W0 / lm.shape[1]), order=0).astype(np.int32)
        sg = res["segs_full"]
        res["segs_full"] = zoom(sg, (H0 / sg.shape[0], W0 / sg.shape[1]), order=0).astype(np.int32)
        res["final_masks"] = [
            zoom(m.astype(np.uint8), (H0 / m.shape[0], W0 / m.shape[1]), order=0).astype(bool)
            for m in res["final_masks"]
        ]
    return res


def _process_image_core(img_np, predictor):
    H, W     = img_np.shape[:2]
    img_area = H * W
    N_SP     = n_superpixels(H, W)

    # Stage 1 — multi-scale SAM2 features + PCA
    feat_map, pca_full, embed_up = extract_features(predictor, img_np)
    edge_map = sobel(img_np.mean(axis=2) / 255.0)

    # Stage 2 — SNIC on embedding space
    labels_list, _, _ = snic(feat_map.tolist(), N_SP, compactness=SNIC_COMPACT)
    segs256   = np.array(labels_list, dtype=np.int32)
    segs_full = zoom(segs256, (H / TARGET, W / TARGET), order=0).astype(np.int32)
    sp_ids    = np.unique(segs_full)

    feat_up = zoom(pca_full, (H / TARGET, W / TARGET, 1), order=1)
    sp_embeddings = {}
    centroids     = {}
    for sid in sp_ids:
        m = (segs_full == sid)
        sp_embeddings[sid] = feat_up[m].mean(0)
        cy, cx = center_of_mass(m)
        centroids[sid] = (float(cx), float(cy))

    # Stage 3 — SAM2 centroid prompts
    valid_masks, valid_scores = [], []
    group_masks = []
    with torch.inference_mode():
        predictor.set_image(img_np)
        for sid in sp_ids:
            cx, cy   = centroids[sid]
            sp_area  = int((segs_full == sid).sum())
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
            bi, bs = select_mask(masks, scores, cx, cy, img_area, sp_area)
            if bi is not None:
                valid_masks.append(clean_mask(masks[bi].astype(bool), img_area))
                valid_scores.append(bs)
            gi = select_group_mask(masks, scores, cx, cy, img_area)
            if gi is not None:
                group_masks.append(clean_mask(masks[gi].astype(bool), img_area))

    # Stage 4 — NMS
    if valid_masks:
        kept        = nms_masks(valid_masks, valid_scores, IOU_THRESH)
        final_masks = [valid_masks[i] for i in kept]
    else:
        final_masks = []

    # Dedup object-level grouping masks (sort by area, drop near-duplicates)
    if group_masks:
        g_areas = [m.sum() for m in group_masks]
        g_kept  = nms_masks(group_masks, g_areas, 0.85)
        group_masks = [group_masks[i] for i in g_kept]

    # Stage 5 — Assign SPs to clubs, track parent mask
    club_label = np.full((H, W), -1, dtype=np.int32)
    sp_parent  = {}
    for obj_idx in np.argsort([-m.sum() for m in final_masks]):
        omask = final_masks[obj_idx]
        for sid in sp_ids:
            sp_mask = (segs_full == sid).astype(bool)
            if float((sp_mask & omask).sum()) / float(sp_mask.sum() + 1e-8) > 0.5 \
               and (club_label[sp_mask] == -1).all():
                club_label[sp_mask] = obj_idx
                sp_parent[sid]      = obj_idx

    # Stage 6 — Unassigned SPs → nearest club by embedding
    assigned = np.unique(club_label[club_label >= 0])
    if len(assigned) > 0:
        club_mean = {}
        for c in assigned:
            members = [s for s in sp_ids if np.any(club_label[segs_full == s] == c)]
            if members:
                v = np.mean([sp_embeddings[s] for s in members], axis=0)
                club_mean[c] = v / (np.linalg.norm(v) + 1e-8)
        ck = list(club_mean.keys())
        cv = np.stack([club_mean[c] for c in ck])
        for sid in sp_ids:
            sp_mask = (segs_full == sid).astype(bool)
            if (club_label[sp_mask] == -1).any():
                emb  = sp_embeddings[sid] / (np.linalg.norm(sp_embeddings[sid]) + 1e-8)
                best = ck[int(np.argmax(cv @ emb))]
                club_label[sp_mask] = best
                sp_parent[sid]      = best
    else:
        for sid in sp_ids:
            club_label[segs_full == sid] = int(sid)
            sp_parent[sid] = int(sid)

    # Stage 7 — Same-parent merge
    club_label = same_parent_merge(club_label, sp_parent, sp_ids, segs_full)

    # Stage 7b — Containment merge (SAM2 object-level grouping)
    if group_masks:
        club_label = containment_merge(club_label, group_masks, embed_up)

    # Stage 7c — Mask-snap (edge-snapping for cluttered scenes)
    if group_masks:
        club_label = mask_snap(club_label, group_masks, SNAP_SOLIDITY, embed_up)

    # Stage 8 — Small-club merge
    club_label = merge_small_clubs(club_label, MIN_CLUB_PX)

    # Stage 9 — Edge-aware embedding merge
    club_embs_tmp = compute_club_embeddings(embed_up, club_label)
    club_label    = edge_aware_embed_merge(club_label, club_embs_tmp, edge_map, EMBED_SIM_THR, EDGE_THR)
    club_label    = merge_small_clubs(club_label, MIN_CLUB_PX)

    club_label = relabel(club_label)

    # Stage 10 — Final clubbed SAM2 embeddings
    club_embs = compute_club_embeddings(embed_up, club_label)

    return {
        "label_map":       club_label,         # (H, W) int32
        "club_embeddings": club_embs,          # {club_id: np.array(256,)}
        "n_clubs":         int(np.unique(club_label).shape[0]),
        "segs_full":       segs_full,          # SNIC superpixel map
        "final_masks":     final_masks,        # SAM2 masks after NMS
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_all(results_list, img_paths, title, out_path):
    n = len(results_list)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, (res, img_path) in enumerate(zip(results_list, img_paths)):
        img_np   = np.array(Image.open(img_path).convert("RGB"))
        club_col = colorise(res["label_map"])
        overlay  = (img_np * 0.55 + club_col * 0.45).clip(0, 255).astype(np.uint8)
        overlay  = mark_boundaries(overlay, res["label_map"], color=(1, 1, 1), mode="thick")
        axes[i].imshow(overlay)
        axes[i].set_title(f"{img_path.name}\n{res['n_clubs']} clubs", fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def visualise_detail(results_list, img_paths, title, out_path, n=4):
    n = min(n, len(results_list))
    fig, axes = plt.subplots(n, 5, figsize=(26, 5.5 * n))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for idx in range(n):
        res     = results_list[idx]
        img_np  = np.array(Image.open(img_paths[idx]).convert("RGB"))
        H, W    = img_np.shape[:2]
        club_col = colorise(res["label_map"])
        overlay  = (img_np * 0.55 + club_col * 0.45).clip(0, 255).astype(np.uint8)
        overlay  = mark_boundaries(overlay, res["label_map"], color=(1, 1, 1), mode="thick")

        ax = axes[idx]
        ax[0].imshow(img_np); ax[0].set_title(f"{img_paths[idx].name}\n{W}×{H}"); ax[0].axis("off")

        snic_vis = mark_boundaries(img_np, res["segs_full"], color=(1, 1, 0), mode="thin")
        ax[1].imshow(snic_vis); ax[1].set_title(f"SNIC on embed"); ax[1].axis("off")

        rng2 = np.random.default_rng(7)
        mv = img_np.copy().astype(float)
        for m in res["final_masks"]:
            col = rng2.integers(80, 255, 3).astype(float)
            mv[m] = mv[m] * 0.3 + col * 0.7
        ax[2].imshow(mv.clip(0, 255).astype(np.uint8))
        ax[2].set_title(f"SAM2 masks ({len(res['final_masks'])})"); ax[2].axis("off")

        ax[3].imshow(mark_boundaries(club_col, res["label_map"], color=(1, 1, 1), mode="thick"))
        ax[3].set_title(f"Clubs ({res['n_clubs']})"); ax[3].axis("off")

        ax[4].imshow(overlay); ax[4].set_title(f"Final overlay"); ax[4].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM2 Superpixel Clubbing Pipeline")
    parser.add_argument("--image_dir",  required=True,  help="Directory of input images")
    parser.add_argument("--output_dir", required=True,  help="Directory to save outputs")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT), help="SAM2 checkpoint path")
    parser.add_argument("--config",     default=CONFIG,           help="SAM2 config name")
    parser.add_argument("--device",     default=None,             help="Device (mps/cuda/cpu)")
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")

    # SAM2 must be imported from its own repo directory
    os.chdir(SAM2_REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"Loading SAM2 on {device}...")
    sam2_model = build_sam2(args.config, args.checkpoint, device=device)
    predictor  = SAM2ImagePredictor(sam2_model)

    img_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    print(f"Found {len(img_paths)} images in '{image_dir.name}'\n")

    all_results  = {}
    results_list = []
    tag = image_dir.name.replace(" ", "_")

    print(f"{'Image':<12} {'Size':>9} {'N_SP':>5} {'Masks':>6} {'Clubs':>6} {'AvgDim':>8}")
    print("-" * 55)

    for idx, img_path in enumerate(img_paths):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W   = img_np.shape[:2]

        res = process_image(img_np, predictor)
        results_list.append(res)

        areas   = [(res["label_map"] == c).sum() for c in np.unique(res["label_map"])]
        avg_dim = float(np.sqrt(np.mean(areas)))

        print(f"[{idx+1:02d}/{len(img_paths)}] {img_path.name:<10} {W}×{H}  "
              f"N_SP={n_superpixels(H,W)}  masks={len(res['final_masks'])}  "
              f"clubs={res['n_clubs']}  avg≈{avg_dim:.0f}px")

        # Strip non-serialisable fields before saving
        all_results[img_path.name] = {
            "label_map":       res["label_map"],
            "club_embeddings": res["club_embeddings"],
            "n_clubs":         res["n_clubs"],
            "meta":            {"file": str(img_path), "size": (W, H),
                                "n_sp": n_superpixels(H, W)},
        }

    # Save embeddings
    emb_path = output_dir / f"{tag}_final.npy"
    np.save(emb_path, all_results)
    print(f"\nEmbeddings saved → {emb_path}")

    # Save visualisations
    all16_path  = output_dir / f"{tag}_final_all.png"
    detail_path = output_dir / f"{tag}_final_detail4.png"

    visualise_all(results_list, img_paths,
                  f"Fine-tuned pipeline — {image_dir.name}", all16_path)
    visualise_detail(results_list, img_paths,
                     f"Detailed — {image_dir.name} (first 4)", detail_path)

    print(f"Grid view  saved → {all16_path}")
    print(f"Detail view saved → {detail_path}")


if __name__ == "__main__":
    main()
