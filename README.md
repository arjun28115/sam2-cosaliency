# SAM2-Only, Training-Free Co-Saliency

Find and segment the object shared by a group of images, using **SAM2 as the only model** — no DINO, CLIP, SigLIP, or trained co-saliency network — and **no training**. All thresholds are self-calibrated from the group's own statistics (percentiles/ranks), so the method is category-agnostic.

SAM2 is used three ways: dense features (partition + matching), a **crop re-encoder** (object-scale descriptors), and the **mask decoder** (final boundaries).

---

## Pipeline

```
A  per image      SAM2 encoder → latent SNIC superpixels (on the 256-d embedding map)
                  → centroid prompts → SAM2 proposals → NMS
                  → subtraction into a partition ("clubs": smallest object wins each pixel)
                  → per-superpixel super-embeddings
B  candidates     every club + every connected union of ≤3 clubs
C  scoring        min-Chamfer set similarity  sim = min(dir(A→B), dir(B→A))
                  evidence = RBD foreground × mutual-consistency eigenvector × robust coverage
D  shortlist      top-K by evidence ∪ top-K by agreement ∪ peer-ranked size strata
E  re-encode      each finalist cropped from the full-res image, re-encoded by SAM2 (raw + masked)
F  clique         pair = rank(qsim)·rank(raw)·rank(masked)·(fg_a·fg_b)^0.25
                  one candidate per image maximising Σpair + λ·evidence (multi-restart ascent)
                  → marginal-gain prune + crop-gated expansion
G  instances      extra non-overlapping instance iff top-3 agreement ≥ P25 of winners' top-3
H  refine         SAM2 decoder on the object crop (mask + padded box + anchor) → LCC + hole fill
```

Detailed walkthroughs: [`reports/PIPELINE_COMPARISON.md`](reports/PIPELINE_COMPARISON.md) and [`reports/REPORT.md`](reports/REPORT.md) (all 43 approaches tried, with measured results).

## Results

| Set | Result |
|---|---|
| Alarm clocks (16) | **16/16** correct, whole-object masks — [figure](figures/FINAL_refined.png) |
| HD airplanes (16 Airbus, 4–6K px) | **16/16** visually, zero-shot, same parameters — [figure](figures/RESULT_airbus16.png) |

**CoCA benchmark** (pixel GT, binary-mask metrics):

| Run | Groups | MAE ↓ | Fβ ↑ | Sα ↑ | Eξ ↑ | IoU ↑ |
|---|---|---|---|---|---|---|
| 12-group dev subset, base config | 12 | 0.119 | 0.512 | 0.659 | – | – |
| 12-group dev subset, final config | 12 | 0.086 | 0.602 | 0.715 | – | – |
| Full run, `final` config (**incomplete**) | 48 / 80 | 0.124 | 0.540 | 0.665 | 0.727 | 0.453 |

Published reference points (full CoCA): ZS-CoSOD (training-free) Fβ 0.549 / Sα 0.667; TF-SSD (training-free, CVPR'26) Fβ 0.686 / Sα 0.763.

**Caveats — read before citing any number above:**
- The full run is **incomplete**: 48 of 80 groups finished, 9 errored (GPU contention from two concurrent runs — retryable), 23 not reached. Resume with `python3 variants/bench.py --all`; finished groups are skipped.
- Published methods report Fβ<sup>max</sup>/Eξ<sup>max</sup> (max over thresholds of a soft saliency map). This pipeline outputs **binary masks**, a single operating point, so the numbers are not directly comparable.
- Oracle analysis: on failing groups the correct object *is* in the candidate set (oracle IoU 0.62–0.83). The remaining gap is **selection**, mainly when scene context recurs more consistently than the object (e.g. hammock, frisbee, dice).

## Repository layout

| Path | Contents |
|---|---|
| `pipeline_final.py`, `snic_latent.py` | **Stage A** (final) |
| `variants/v_final.py` | **Stages C–F** (final method) |
| `variants/v_base.py`, `v_combo.py`, `v_reencode.py`, `v_anneal.py`, `harness.py` | helpers used by the final method |
| `variants/multi_instance.py`, `refine_masks.py` | **Stages G, H** |
| `variants/run_dataset.py` | one command: folder → selection → instances → refined masks → figure |
| `variants/bench.py`, `metrics.py` | CoCA benchmark harness (MAE / Fβ / Sα / Eξ / IoU) |
| `variants/bench_results.json` | per-group metrics for every config (`base`, `it3_fgclique`, `it4_fgsoft`, `it5_rank`, `final`) |
| `variants/gt_*.py` | hand-verified GT boxes for the clock / dice / clover sets |
| `variants/v_profile.py`, `extent_snap.py`, … | measured negative results, kept for the record |
| top-level `co_saliency*.py`, `cosal_*.py`, `run_*.py`, `club_ips*.py`, `pipeline_spam*.py`, … | earlier explorations (matrix/per-dimension consensus, spectral, coverage, set, union, etc.) |
| `semantic_clubs.py`, `club_ips_dino.py` | SigLIP / DINO experiments — **not** part of the SAM2-only method |
| `reports/` | full report, pipeline comparison, explained-pipeline PDFs |
| `figures/` | curated result figures |
| `patches/build_sam.patch` | small patch to SAM2's `build_sam.py` (see Setup) |

## Setup

1. Clone SAM2 and download the large checkpoint:
   ```bash
   git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .
   ```
   Put `sam2.1_hiera_large.pt` in `sam2/checkpoints/`.
2. Copy this repository's files into the root of that clone (scripts expect `pipeline_final.py` next to the `sam2/` package and `variants/` beside it).
3. Apply the patch, which disables SAM2's "running from the repo parent directory" guard:
   ```bash
   git apply patches/build_sam.patch
   ```
4. Run:
   ```bash
   python3 variants/run_dataset.py "/path/to/image_folder" mytag
   ```

**Note:** several scripts contain hard-coded absolute paths (`/Users/arjunsingla/...`) for data, GT, and cache locations. Edit those to your own paths before running. Model weights, cached arrays, and the CoCA images/GT are not included (size and licensing).

Developed on an Apple Silicon Mac (MPS).
