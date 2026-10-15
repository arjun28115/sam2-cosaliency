# SAM2-Only Training-Free Co-Saliency — Full Project Report
*Everything tried, what worked, what didn't — with measured numbers. (as of 2026-07-03)*

---

## 1. Goal and constraints

Find the common (co-salient) object across a group of images and segment it, using **SAM2 as the only model** (no DINO, SigLIP, CLIP, or any trained CoSOD network), fully **training-free**, with **category-agnostic** math (no constants tuned to a specific object class; scale-free percentile/rank thresholds only).

## 2. Base per-image pipeline (fixed structure throughout)

```
image (any res; logic at WORK_MAXSIDE=1280)
→ SAM2 encoder → image_embed (64×64×256)
→ latent SNIC superpixels (clustering directly on the 256-d embedding map)
→ prompts (superpixel centroids, or uniform grid mode)
→ SAM2 object proposals → NMS
→ subtraction into a partition: "clubs" (smallest object wins each pixel)
→ super-embeddings: mean image_embed per superpixel (L2-normed)
→ clubbed super-embeddings (size-weighted mean per club)
```

Earlier pipeline work (pre co-saliency): bleed-mask rejection (>25% of image), two-clock merge fix (03.jpg), bag+clock and smiley+clock separation (01/12.jpg), 4K/HD capability, latent SNIC replacing PCA+pysnic, SPAM assignment variants.

## 3. Co-saliency methods tried — chronological ledger

| # | Method / idea | Result (clock set) | Verdict |
|---|---|---|---|
| 1 | Inter-image consensus A–D (L2-norm → P·Pᵀ → max per other image → mean) | ~13/16 | baseline; background-prone |
| 2 | Background border penalty (Tip #1) + Otsu/top-k masks + vectorized matmul | ~13/16 | kept as stage-1 pieces |
| 3 | TF-SSD / IPS scoring (paper) | — | discarded ("forget IPS") |
| 4 | max instead of mean aggregation | ~13/16 | no change in ceiling |
| 5 | N×N block-matrix formulation | identical to flat (diff 2e-7) | equivalent, kept as viz |
| 6 | Greyscale-image variant | same ceiling | no advantage |
| 7 | Group-center proxy Fc + top-k prompt points (CoSOD paper) | ~13–14/16 | complementary errors |
| 8 | Ensemble consensus × proxy (soft-AND) | ~14/16 | kept idea |
| 9 | SAM2 refinement prompts: point / negative point / mask-prompt / mask+anchor | cleaner masks | mask+anchor best |
| 10 | Trained CoSOD (VGG16/ConvLSTM paper) | — | rejected (training-free constraint) |
| 11 | RBD boundary connectivity saliency | coherence 0.588→0.713, ~14/16 | **adopted** |
| 12 | Combined consensus × RBD × proxy → SAM2 refine | ~14–15/16 | best of the early era |
| 13 | Dual-stream fusion with SigLIP (score-level / concat) | ~16/16 but **violates constraint** | rejected by constraint |
| 14 | Per-dimension decomposition (256 per-channel N×N matrices, pooled back to 256-d) | 11–13/16 | **worse** — max(Σ)≤Σ(max): decoupling channels breaks same-object matching |
| 15 | Spectral consensus (principal eigenvector) + RBD foreground | 13/16 | kills backgrounds |
| 16 | Coverage / ubiquity (mean of worst-k per-image matches) | 12/16 alone | re-admits backgrounds alone |
| 17 | fg × eigenvector × coverage combined | 12–14/16 | mug/teddy persist |
| 18 | Set-based min-Chamfer similarity (superpixel sets, min of two directions) | 14/16 | **kills uniform distractors** (cup, books) |
| 19 | Union candidates (connected club unions ≤3) | 15/16 | **fixes fragmented objects** |
| 20 | Consensus assignment (coordinate ascent), gated, global clique + restarts | 14–15/16 | whack-a-mole among {00,07,11,14}; measured inversion: pillow 0.566 vs clock 0.550 |
| 21 | Geometric layout consistency (part-position correlation) | pillow 0.54 > clock 0.35 | **refuted by measurement**, discarded |
| 22 | Orphan-club repair (label-map regions with no superpixel mapping) | 12→13 on color data | **bug fix, adopted** |
| 23 | Second-order profile similarity (candidates match the same things) | ≤13/16 | failed — profiles dominated by generic signal |
| 24 | tf-idf genericness weighting (multiplicity-based superpixel weights) | no change | no effect at any β |
| 25 | Union internal-coherence gate | broke imgs 0/4/6 | destructive — real objects ARE dark+light unions |
| 26 | Marginal-gain prune of picked unions | fixed img14 → 14/16 | **adopted** |
| 27 | Annealed quantile agreement (worst-parts must match) | fixed 11,14; broke 00 | circular; hysteresis window only 0.986–1.012, too thin |
| 28 | Frankenstein-union discovery (mug+blanket, pillow+sheet mimic dark-body+white-face) | diagnosis | key insight driving #29 |
| 29 | **SAM2 crop re-encoding** of finalists (raw + masked, object scale) | raw 14/16, masked+ev 15/16 | **the ceiling-breaker** |
| 30 | **Soft-AND consensus clique**: nrm(qsim)×nrm(raw)×nrm(masked)+λ·evidence | **16/16** | **FINAL METHOD** — ablation: drop any view → 14/16 |
| 31 | Robustness sweep | 16/16 for q∈{0.2,0.3,0.5} at λ=0.15 | λ is the one balanced knob |
| 32 | Multi-scale super-embeddings (fuse high_res_feats 256²+128²) | clocks 16→13 | **regression, reverted** — hr feats are structural, not semantic |
| 33 | Multi-instance, relative threshold | pillow 102% > white clock 93% | failed — weak winners inflate ratios |
| 34 | Multi-instance, absolute P25 bar (mean agreement) | white clock in; dotted mug also in (0.463 vs bar 0.456) | 1 borderline false positive |
| 35 | **Multi-instance, top-3 near-duplicate rule** (P25 of winners' top-3) | white clock 0.627 ✓, mug 0.521 ✗ (bar 0.550) | **adopted** — 20% margin |
| 36 | Mask refinement: mask+box+anchor decoder prompt | crisp masks | adopted |
| 37 | **Zoom refinement** (decode on object crop; padded box) | bells/gear recovered; HD-sharp | **adopted** |
| 38 | Largest-connected-component + hole fill | speckles removed | adopted (cosmetic) |
| 39 | Grid prompts + MIN_AREA_FRAC=0.0005 for tiny objects | dice reachability 12→13/16 | pipeline dial, adopted for tiny sets |
| 40 | Fine superpixels (BASE_N_SP 600) | recall unchanged | not the binding constraint |
| 41 | Size-stratified finalist shortlist (small_k) | clover/dice unchanged; clock 16/16 held | kept (harmless, principled) |
| 42 | Top-3 crop gate in expansion | clover unchanged | kept; no effect measured |
| 43 | Extent snapping (joint coordinate descent on pick extents, top-3 crops) | dice 2→2, clover 5→5, clock 16→16 | **no net effect**; produced the context-near-duplicate diagnosis |

## 4. Final method (v_final.py, K=10, q=0.3, λ=0.15)

1. Candidates: clubs + connected unions ≤3 (orphan-repaired)
2. Stage-1: min-Chamfer set similarity; evidence = RBD × eigenvector × robust coverage; gated assignment
3. Finalists: top-10 by evidence ∪ top-10 by agreement ∪ top-5 small-stratum
4. SAM2 crop re-encoding of every finalist (raw + masked) → object-centric 256-d
5. Soft-AND consensus clique + multi-restart coordinate ascent
6. Marginal-gain prune + crop-gated expansion
7. Multi-instance: top-3 agreement ≥ P25(winners' top-3), non-overlapping
8. Zoom mask refinement (SAM2 decoder on object crop, padded box, LCC+fill)

## 5. Evaluation protocol

Hand-annotated GT boxes per image (verified visually); hit requires **union-purity ≥ 0.6** (pick lies on GT objects) AND **coverage ≥ 0.25** of at least one instance box. Plus visual verification of every reported number. Known criterion pitfall found and fixed: cluster boxes for sparse multi-instance sets made perfect picks "unreachable" (dice).

## 6. Results

| Dataset | Strict score | Visual | Candidate-space ceiling | Notes |
|---|---|---|---|---|
| Alarm clocks (16) | **16/16** | 16/16 | 16/16 | + 2nd instance in img03, no false extras |
| Airplanes HD (16 of 586 Airbus) | not box-annotated | **16/16** | — | zero-shot, frozen params, ~25 min |
| Clover (10) | 5/10 | ~8/10 on-object | 9/10 | partial-leaf picks; img3 unreachable |
| Dice (16) | 2/16 | ~4/16 | 13/16 (grid prompts) | picks contain dice but absorb context |

## 7. Characterized ceilings (measured, not speculated)

1. **Appearance ≠ category**: SAM2 embeddings encode appearance. Where a look-alike matches the group better than a weak true instance (pillow 0.566 vs clock 0.550), no training-free inference re-orders it.
2. **Dice / context co-recurrence**: the group's *surfaces* (felt, boards, cards) are near-duplicates across images (top-3 crop agreements 0.87–0.95) — preferring dice over surfaces is a semantic judgment appearance cannot make.
3. **Clover / heterogeneous subgroups**: dark-center vs pure-green clovers → consensus keeps only the universally common part; partial picks validate each other (self-consistent wrong extent). Five independent mechanisms failed identically.
4. **Tiny-object recall**: objects ≤ ~1–2% of image area dissolve in the proposal→partition stage; grid prompts + small min-area recover some (dice 12→13 reachable), motion-blurred / sub-40px objects stay unreachable.
5. **Decoder resolution**: fixed by zoom refinement (decode on crop); residual limit is SAM2's 1024-px internal budget per crop.

## 8. What is honestly achieved vs not

**Achieved:** clocks 16/16 (strict + visual, regression-proof), airplanes 16/16 (visual, zero-shot), principled multi-instance recovery, HD-crisp masks, no hand constants, SAM2-only throughout, one-command runner (`variants/run_dataset.py`).

**Not achieved:** clover beyond 5/10 strict, dice beyond 2/16, clock-07 selection purity (blanket club), standard benchmark numbers (CoCA/CoSOD3k never run — no SOTA claim possible), group-size scaling beyond N≈16 (O(N²)).

## 9. Assets

- Code: `/Users/arjunsingla/sam2/` (pipeline_final.py, snic_latent.py) and `/Users/arjunsingla/sam2/variants/` (harness, v_base, v_combo, v_anneal, v_reencode, v_final, multi_instance, refine_masks, extent_snap, run_dataset, step_run, gt_color/gt_dice/gt_clover)
- Data: `/Users/arjunsingla/sam2/data/*.npy` (durable; /tmp gets wiped)
- Renders: `/Users/arjunsingla/Desktop/SAM2_Pipeline/cosaliency/` (FINAL_16of16, FINAL_refined, RESULT_airbus16, GENERAL_clover, GENERAL_dice2, gt box overlays, diagnostics)

## 10. Open next steps

1. CoCA / CoSOD3k benchmark harness → real standing vs published training-free methods
2. Dice/clover ceilings: would require semantic signal (excluded by constraint) or a group-prior beyond appearance
3. Batch runner for the remaining 570 Airbus images; mixed-type group stress test (Airbus+F16+C130)
