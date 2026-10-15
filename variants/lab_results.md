# Selection-lab results log

Coarse partition masks (no SAM2 refinement), so absolute values are a proxy; compare rows within a table.

## 2026-10-15 — selection sweep, 8 groups (mean IoU over groups)
| config | IoU | F |
|---|---|---|
| baseline (current clique: views q,r,m; λ=0.15; fg^0.25) | 0.497 | 0.590 |
| robust aggregation, worst 30% | 0.442 | 0.542 |
| robust aggregation, worst 50% | 0.476 | 0.572 |
| + distinctiveness 0.3 / 0.6 | 0.498 / 0.496 | 0.591 / 0.590 |
| λ = 0 | 0.513 | 0.607 |
| λ = 0.4 | 0.485 | 0.579 |
| fg^0 / fg^0.5 / fg^1.0 | 0.506 / 0.496 / 0.499 | 0.600 / 0.589 / 0.592 |
| views q,m | 0.497 | 0.592 |
| **views r,m** | **0.528** | **0.625** |
| views m | 0.514 | 0.612 |
| oracle (best finalist per image) | ~0.67 | – |

## 2026-10-15 — standard protocol (dataset-level Fβmax/Eξmax), 16 groups, 256 images
| config | Fβmax | Eξmax |
|---|---|---|
| current method, binary | 0.604 | 0.768 |
| current method, soft map | 0.610 | 0.768 |
| views r,m, λ=0, binary | 0.648 | 0.792 |
| views r,m, λ=0, soft γ=2 / 4 / 1 | 0.648 / 0.648 / 0.647 | 0.793 / 0.794 / 0.793 |

Reference (full CoCA): TF-SSD Fβmax 0.686, ZS-CoSOD 0.549.

## Images
`figures/lab/lab_wins.png`, `figures/lab/lab_losses.png` — current method vs. views (r,m), λ=0.
Picks differ on 146 / 256 images: better on 46, worse on 43. Wins = context-absorption fixed
(camera, beer bottle, duck, binoculars found instead of suitcase/tray/gull/person). Losses =
part-instead-of-whole (camera lens only, butterfly + flower) and wrong object (books for clock).

## Status when paused (2026-10-15) — how to resume
- Lab caches (`data/coca/lab_<group>.npz`): 44 / 80 groups built. Resume: `./lab_driver.sh` (skips done groups).
- Deep SAM2 trunk features (`trunk_<group>.npz`): only `alarm_clock` built, not yet evaluated.
  Resume: `python3 lab_trunk.py Accordion UAV Yellow_duck avocado backpack baseball clover`.
- Next experiments:
  1. crop views (r,m) choose WHICH object, patch view (q) decides extent — target the part-vs-whole losses;
  2. trunk-feature views (t3m/t4m) in `pair_matrix` on the 8 dev groups;
  3. best config -> full 80-group run with SAM2 refinement + soft maps, scored by `metrics_std.py`.
- Current best proxy: Fβmax 0.648 / Eξmax 0.792 (16 groups, coarse masks). TF-SSD: 0.686 (full CoCA).
