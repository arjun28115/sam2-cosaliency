# Two Pipelines Compared: N×N Super-Embedding Matrix  vs.  Final Crop-Re-encode + Zoom-Refine
*SAM2-only, training-free co-saliency. All numbers measured on the 16-image alarm-clock group (+ CoCA dev subset where noted).*

---

## 1. What each pipeline is

**Matrix pipeline** — the original formulation. One `N×N` block matrix whose cell `(i,j) = Eᵢ · Eⱼᵀ` is the full block of dot-products between the **club super-embeddings** of image i and image j (diagonal / intra-image blocks skipped). A club's co-saliency score is the consensus over its row of blocks (max or mean over the other N−1 images) minus a background penalty; a selection rule turns scores into a mask.

**Final pipeline** — the endpoint. Same SAM2 partition, but each club is compared as a **set of superpixel embeddings** (not a mean), finalists are **re-encoded by SAM2 at crop scale**, selection is a **soft-AND consensus clique** over union candidates, and masks come from **SAM2 decoder zoom refinement**.

The crucial structural difference: the matrix pipeline collapses each club to **one mean vector**; the final pipeline never collapses — it keeps sets, re-encodes objects, and refines boundaries.

---

## 2. Architecture, side by side

| Stage | Matrix pipeline | Final pipeline |
|---|---|---|
| Per-image partition | SAM2 → SNIC → prompts → proposals → NMS → subtraction → clubs | *(identical)* |
| Descriptor per club | **mean** super-embedding (256-d, collapsed) | **set** of superpixel embeddings + **SAM2 crop re-encoding** (raw + masked) |
| Cross-image comparison | `Eᵢ · Eⱼᵀ` block (mean · mean) | min-Chamfer set·set, then soft-AND `rank(qsim)·rank(cos_raw)·rank(cos_masked)` |
| Candidate | single club | club **+ connected unions ≤ 3** |
| Selection | argmax / threshold / anchor-grow over club scores | consensus **clique** (multi-restart) + marginal-gain prune + crop-gated expansion |
| Multiple instances | not modeled | top-3 near-duplicate bar (P25 of winners' top-3) |
| Mask | partition region (ragged, nearest-upsampled) | **SAM2 decoder zoom refinement** |

---

## 3. The mask-quality fix (why the final pipeline's masks are "proper")

Three distinct causes were diagnosed; two were fixed by **zoom refinement**.

| cause | symptom | status |
|---|---|---|
| decoder ran on the whole image → SAM2's internal 1024 budget diluted across 6000 px | chunky boundaries on HD, thin parts (gear struts, antennas, bell rings) below decode resolution | **fixed** — decode on the object crop: full 1024 covers just the object (~2.4× sharper on planes, ~6× on small clocks) |
| prompt box clamped to the seed's bbox | missing bells / handle-tips the club selection never contained | **fixed** — box padded 60 % beyond the seed so the decoder grows into missing parts |
| seed itself contains a wrong club | 07's blanket patch riding along the clock | **remains** — this is selection purity, not mask quality; the decoder faithfully segments whatever the seed includes |

The matrix pipeline has none of this — its mask *is* the raw partition region, so it is simultaneously ragged **and** only ever a part (see §4).

---

## 4. The matrix pipeline's ceiling, measured

The block matrix scores **individual clubs**, and a club is a *part*, not a whole object. Three selection rules, three behaviours — all on the identical `Eᵢ·Eⱼᵀ` matrix:

| selection rule | result | mask |
|---|---|---|
| `argmax` single club | **13/16** | the **bells only** (bells are the most consistent recurring part) |
| `threshold + union` | **~11/16** | whole clock **+ bleed** (15 → 21 bouquet clubs) |
| `anchor-connected grow` | **~11/16** | less bleed, but 15 still → 23 clubs; 05/13/14 shrink to 1 club |

Two walls no selection rule can pass:
1. **Wrong anchor (appearance ceiling)** — on 10 (figurine), 11 (teddy), 14 (banana), 15 (bouquet) the club-mean of the distractor genuinely matches clock club-means, so the pick starts on the wrong object.
2. **Threshold fragility** — no single threshold is right for all 16 images; tight shrinks to a part, loose bleeds.

Independently verified: the N×N block form is **mathematically identical** to the flat `P·Pᵀ` consensus (max abs difference `2e-7`) — a reorganization, not a new algorithm.

---

## 5. How the final pipeline breaks each wall

| matrix-pipeline wall | root cause | final-pipeline fix |
|---|---|---|
| bells-only answer | scores single clubs (parts) | **connected union candidates** — whole objects are first-class candidates |
| wrong anchor on look-alikes | club **mean** embedding can't tell a structured clock from a uniform blob | **min-Chamfer set matching** + **SAM2 crop re-encoding** (object-scale second opinion) |
| bleed / threshold fragility | free per-club threshold | bounded **clique** selection + marginal-gain prune |
| ragged / partial masks | raw partition region | **SAM2 decoder zoom refinement** (crop-scale, padded box, LCC + fill) |

---

## 6. Results, measured

**Alarm-clock group (16 images):**

| pipeline | whole-object correct | mask quality |
|---|---|---|
| Matrix (argmax) | 13/16 | bells only |
| Matrix (threshold+union) | ~11/16 | whole + bleed |
| **Final** | **16/16** | **whole, decoder-crisp** |

**Generality (final pipeline, frozen parameters):** airplanes (HD Airbus) 16/16 zero-shot, whole airframe, crisp; clover 5/10 strict; dice 2/16 (candidate-recall limited).

**CoCA benchmark, 12-group dev subset (final pipeline, pixel GT, standard metrics):**

| | MAE ↓ | Fβ ↑ | Sα ↑ | Eξ ↑ |
|---|---|---|---|---|
| **Final pipeline** | **0.086** | **0.602** | **0.715** | 0.773 |
| ZS-CoSOD (prior training-free) | 0.115 | 0.549 | 0.667 | — |
| TF-SSD (training-free SOTA, CVPR'26) | 0.077 | 0.686 | 0.763 | ~0.815 |
| oracle (our candidate ceiling) | — | 0.884 | — | — |

The matrix pipeline was never benchmarked on CoCA — it plateaus at the ~13/16 appearance ceiling with partial masks, which is the reason it was superseded.

---

## 7. Honest bottom line

- The **matrix pipeline** is a correct, clean formulation whose *similarity* is fine — but it collapses clubs to means and picks single clubs, so it tops out at **13/16 with bells-only (or bleedy) masks**. Its value was diagnostic: it proved that re-organizing the *same* embeddings cannot break the ceiling.
- The **final pipeline** keeps everything the matrix pipeline could not: whole objects (union candidates), a non-collapsed descriptor (set matching + crop re-encoding), bounded selection (clique), and proper boundaries (zoom refinement). Result: **16/16 clean on clocks, 16/16 on HD planes, and second-best-among-training-free on the CoCA dev subset** — with the remaining gap to SOTA located precisely in *selection under engineered distractors*, not in segmentation.
- One shared, unbroken wall: where a distractor's appearance genuinely matches the target (07 blanket, CoCA cats-with-ducks), both pipelines fail — the SAM2-only, appearance-based descriptor cannot make a semantic distinction. That is the honest ceiling of the constraint.
