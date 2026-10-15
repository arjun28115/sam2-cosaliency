"""
Geometric ground truth for the color clock set: normalized bounding boxes
(x0, y0, x1, y1) around every real alarm clock in each image. Multiple boxes
when an image contains several clocks (any of them counts).

hit criterion (see harness.evaluate_boxes):
  purity   : >= 60% of picked mask pixels fall inside a clock box
  coverage : picked mask covers >= 25% of that box's area
(reflections, look-alikes and background fail purity; a single bell or a
sliver of the face fails coverage)
"""

BOXES = {
    0:  [(0.53, 0.70, 0.73, 0.97)],
    1:  [(0.50, 0.22, 0.97, 0.95)],
    2:  [(0.38, 0.05, 0.88, 0.98)],
    3:  [(0.08, 0.15, 0.38, 0.75), (0.55, 0.14, 0.85, 0.72)],
    4:  [(0.10, 0.05, 0.75, 0.56)],
    5:  [(0.33, 0.30, 0.62, 0.80)],
    6:  [(0.05, 0.15, 0.58, 0.95)],
    7:  [(0.15, 0.55, 0.42, 0.90)],
    8:  [(0.10, 0.15, 0.45, 0.85)],
    9:  [(0.52, 0.10, 0.95, 0.90)],
    10: [(0.03, 0.15, 0.42, 0.85)],
    11: [(0.02, 0.30, 0.25, 0.80)],
    12: [(0.15, 0.25, 0.55, 0.95)],
    13: [(0.15, 0.35, 0.35, 0.65)],
    14: [(0.60, 0.45, 0.92, 0.90)],
    15: [(0.02, 0.05, 0.55, 0.95)],
}
