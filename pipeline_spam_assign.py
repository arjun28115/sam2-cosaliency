"""
SPAM pipeline — SUPER-EMBEDDING-CONSTRAINED variant
---------------------------------------------------
Same as pipeline_spam.py, EXCEPT the prior constraint is applied at the
super-embedding level instead of the superpixel geometry:

  pipeline_spam.py  (geometry):  split each SNIC superpixel at object boundaries,
                                 then embed each piece -> embeddings are boundary-
                                 pure, clubbed == pixel-based (cos ~ 1.0), but the
                                 superpixel count grows (extra boundary slivers).

  THIS FILE         (assign):    keep raw SNIC superpixels intact, compute ONE
                                 256-d super-embedding per raw superpixel, then
                                 ASSIGN each superpixel to its dominant club
                                 (the club covering most of it). No splitting ->
                                 a clean 1:1 set (N super-embeddings == N SNIC
                                 superpixels), at the cost of a small boundary
                                 blur in the clubbed embeddings (cos ~ 0.98).

Use this when you want a fixed-size, one-node-per-superpixel representation
(e.g. graph / transformer input). Use pipeline_spam.py when you need the clubbed
embeddings to faithfully equal the pixel-level object embeddings.

This file does NOT modify pipeline_spam.py — it imports and reuses its functions.

Usage:
    python pipeline_spam_assign.py --image_dir /path/to/images --output_dir /path/to/out
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

import pipeline_spam as base   # reuse all shared building blocks; never edited


def super_embeddings_assign(predictor, segs, prior):
    """
    Constrain at the super-embedding level (no geometric splitting):
      - one 256-d super-embedding per RAW SNIC superpixel (mean image_embed)
      - assign each superpixel to its dominant club (majority pixel overlap)
    Returns: ids, matrix (N,256), sizes (N,), sp_to_club {id: club}.
    """
    H, W = segs.shape
    embed_up = base._image_embed_upsampled(predictor, H, W)
    ids = np.unique(segs)
    mat = np.zeros((len(ids), 256), dtype=np.float32)
    sizes = np.zeros(len(ids), dtype=np.int64)
    sp_to_club = {}
    for i, sid in enumerate(ids):
        m = (segs == sid)
        v = embed_up[m].mean(0)
        mat[i] = v / (np.linalg.norm(v) + 1e-8)
        sizes[i] = int(m.sum())
        vals, cnts = np.unique(prior[m], return_counts=True)   # dominant club
        sp_to_club[int(sid)] = int(vals[cnts.argmax()])
    return ids, mat, sizes, sp_to_club


def process_image(img_full, predictor):
    H0, W0 = img_full.shape[:2]
    scale = base.WORK_MAXSIDE / max(H0, W0)
    if scale < 1.0:
        work = np.array(Image.fromarray(img_full).resize(
            (max(1, int(W0 * scale)), max(1, int(H0 * scale))), Image.BILINEAR))
    else:
        work = img_full
    H, W = work.shape[:2]
    img_area = H * W

    with torch.inference_mode():
        predictor.set_image(work)

    # 1. SNIC superpixels on SAM2 embeddings (kept RAW — not split)
    segs = base.snic_superpixels(predictor, H, W)

    # 2. SAM2 object proposals (centroid or grid prompts)
    pts = base.superpixel_centroids(segs) if base.PROMPT_MODE == "centroid" \
        else base.grid_points(H, W, base.GRID)
    proposals = base.nms(base.sam_point_proposals(predictor, pts, img_area), base.IOU_THRESH)

    # 3. subtraction prior -> object clubs
    prior, _ = base.aggregate(proposals, H, W)

    # 4. super-embeddings + ASSIGN constraint (no splitting)
    sp_ids, sp_mat, sp_sizes, sp_to_club = super_embeddings_assign(predictor, segs, prior)

    # 5. club the super-embeddings (size-weighted mean per club)
    club_super = base.club_from_super(sp_ids, sp_mat, sp_sizes, sp_to_club)
    club_pixel = base.region_embeddings(base._image_embed_upsampled(predictor, H, W), prior)

    if scale < 1.0:
        prior = zoom(prior, (H0 / H, W0 / W), order=0).astype(np.int32)
        segs  = zoom(segs,  (H0 / H, W0 / W), order=0).astype(np.int32)

    return {
        "label_map": prior,                       # object clubs
        "superpixels": segs,                      # RAW SNIC superpixels (un-split)
        "superpixel_ids": sp_ids,                 # (N,)
        "super_embeddings": sp_mat,               # (N, 256) one per superpixel
        "sp_to_club": sp_to_club,                 # {superpixel id: club id}
        "club_super_embeddings": club_super,      # {club: (256,)} assign-clubbed
        "club_embeddings": club_pixel,            # {club: (256,)} pixel-based reference
        "n_clubs": int(np.unique(prior).shape[0]),
        "n_superpixels": int(len(sp_ids)),
        "n_proposals": len(proposals),
    }


def main():
    ap = argparse.ArgumentParser(description="SPAM pipeline (super-embedding-constrained variant)")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--checkpoint", default=str(base.CHECKPOINT))
    ap.add_argument("--config", default=base.CONFIG)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(base.SAM2_REPO)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(f"Loading SAM2 ({Path(args.checkpoint).name}) on {device}...")
    predictor = SAM2ImagePredictor(build_sam2(args.config, args.checkpoint, device=device))

    img_dir = Path(args.image_dir)
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    tag = img_dir.name.replace(" ", "_")
    print(f"Found {len(paths)} images in '{img_dir.name}'  [assign-constrained variant]\n")

    results, store = [], {}
    print(f"{'image':<14}{'clubs':>6}{'superpix':>10}{'cos(club~pix)':>15}")
    print("-" * 45)
    for i, pth in enumerate(paths):
        img = np.array(Image.open(pth).convert("RGB"))
        res = process_image(img, predictor)
        results.append(res)
        cs, cb = res["club_super_embeddings"], res["club_embeddings"]
        cos = float(np.mean([cs[c] @ cb[c] for c in cs if c in cb]))
        store[pth.name] = {
            "label_map": res["label_map"], "superpixels": res["superpixels"],
            "superpixel_ids": res["superpixel_ids"], "super_embeddings": res["super_embeddings"],
            "sp_to_club": res["sp_to_club"], "club_super_embeddings": cs, "club_embeddings": cb,
            "n_clubs": res["n_clubs"], "n_superpixels": res["n_superpixels"],
            "meta": {"file": str(pth), "size": img.shape[:2][::-1]},
        }
        print(f"[{i+1:02d}/{len(paths)}] {pth.name:<8}{res['n_clubs']:>6}{res['n_superpixels']:>10}{cos:>15.4f}")

    np.save(out_dir / f"{tag}_spam_assign.npy", store)
    base.visualise(results, paths, f"SPAM (assign-constrained) — {img_dir.name}",
                   out_dir / f"{tag}_spam_assign.png")
    print(f"\nSaved → {out_dir / (tag + '_spam_assign.npy')}")
    print(f"Saved → {out_dir / (tag + '_spam_assign.png')}")


if __name__ == "__main__":
    main()
