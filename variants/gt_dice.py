"""Geometric GT for the dice set: TIGHT normalized (x0,y0,x1,y1) boxes around
individual dice / compact dice groups. Evaluation uses union-purity (pick must
lie on dice) + best-single-box coverage (pick must substantially cover at
least one die)."""

BOXES = {
    0:  [(0.49, 0.27, 0.61, 0.47)],                            # dice on the face
    1:  [(0.54, 0.33, 0.68, 0.52)],                            # dice at the hand
    2:  [(0.41, 0.42, 0.56, 0.63), (0.53, 0.36, 0.66, 0.55),
         (0.62, 0.32, 0.73, 0.47)],                            # three dice in palm
    3:  [(0.36, 0.50, 0.50, 0.72), (0.48, 0.42, 0.60, 0.62),
         (0.52, 0.55, 0.68, 0.78)],                            # dice by leather cup
    4:  [(0.09, 0.24, 0.18, 0.39), (0.19, 0.27, 0.28, 0.42),
         (0.32, 0.20, 0.41, 0.34), (0.29, 0.40, 0.44, 0.63)],  # red dice, poker table
    5:  [(0.34, 0.20, 0.68, 0.78)],                            # big red die
    6:  [(0.27, 0.36, 0.37, 0.52), (0.39, 0.64, 0.49, 0.80)],  # red + purple dice
    7:  [(0.60, 0.52, 0.74, 0.70)],                            # dice pair, backgammon
    8:  [(0.60, 0.44, 0.76, 0.64)],                            # dice among mahjong
    9:  [(0.15, 0.60, 0.29, 0.82)],                            # blue-pip die
    10: [(0.33, 0.58, 0.42, 0.70), (0.41, 0.53, 0.49, 0.64),
         (0.44, 0.62, 0.53, 0.74), (0.51, 0.55, 0.59, 0.67),
         (0.47, 0.50, 0.55, 0.60)],                            # five dice in tray
    11: [(0.52, 0.54, 0.61, 0.68), (0.62, 0.56, 0.71, 0.70),
         (0.70, 0.48, 0.79, 0.62), (0.43, 0.34, 0.53, 0.47)],  # small dice, pink cloth
    12: [(0.50, 0.40, 0.79, 0.84)],                            # red dice pair
    13: [(0.43, 0.26, 0.56, 0.46), (0.51, 0.38, 0.65, 0.58)],  # b/w dice pair
    14: [(0.11, 0.62, 0.26, 0.80)],                            # white dice pair
    15: [(0.79, 0.22, 0.95, 0.50)],                            # fuzzy die on mirror
}
