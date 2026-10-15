"""
Shared evaluation for co-saliency variants on /tmp/clock_grey.npy.
GT = acceptable clock club ids per image (any club that is part of an actual
alarm clock in that image, verified visually across earlier renders).

hit(image) requires BOTH:
  coverage : the picked mask contains at least one acceptable clock club
  purity   : >= 50% of picked mask pixels lie on acceptable clock clubs
             (so "clock + huge junk union" doesn't count as a hit)
"""
import numpy as np

# acceptable clock clubs per image index (grey data)
GT = {
    0: {11},          # small blue clock on glass table
    1: {7, 30},       # red clock (body, small part)
    2: {4},           # red clock
    3: {4, 8, 10},    # black clock (face/body) + white clock
    4: {8, 10},       # red clock body + white face
    5: {4, 6, 10},    # black clock parts
    6: {6},           # orange clock
    7: {5},           # small red clock in blanket
    8: {7, 9, 11},    # black clock face/body/bells
    9: {4, 6},        # black clock
    10: {6, 7},       # gold clock
    11: {9},          # dim clock left
    12: {1, 2},       # red clock
    13: {18},         # small travel clock
    14: {3, 5},       # white clock
    15: {5},          # white clock face
}


def evaluate(picks, data, purity_thresh=0.5, verbose=False):
    """picks: list of (member_club_ids, mask HxW bool) per image. Returns (hits, misses)."""
    hits, misses = [], []
    for n, (mem, mask) in enumerate(picks):
        mem = set(int(c) for c in mem)
        lm = data[n]["label_map"]
        acc = GT[n]
        has = len(mem & acc) > 0
        if mask.sum() == 0:
            purity = 0.0
        else:
            acc_mask = np.isin(lm, list(acc))
            purity = float((mask & acc_mask).sum()) / float(mask.sum())
        ok = has and purity >= purity_thresh
        (hits if ok else misses).append(n)
        if verbose:
            print(f"  {n:02d}: mem={sorted(mem)} has_clock={has} purity={purity:.2f} -> {'HIT' if ok else 'MISS'}")
    return hits, misses


def load_grey():
    return np.load("/tmp/clock_grey.npy", allow_pickle=True)
