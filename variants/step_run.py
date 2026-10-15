"""One refinement step: run v_final on a dataset, evaluate, print one line.
Usage: python3 step_run.py <data.npy> <gt_module|none> <cache_dir> <small_k> [tag]"""
import sys
from pathlib import Path
sys.path.insert(0, "/Users/arjunsingla/sam2/variants")

data_path, gt_mod, cache_dir, small_k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tag = sys.argv[5] if len(sys.argv) > 5 else Path(data_path).stem

import v_reencode as vr
vr.CACHE_DIR = Path(cache_dir)
vr.CACHE_DIR.mkdir(exist_ok=True)

from harness import load, evaluate
from v_final import run

data = load(data_path)
picks = run(data, K=10, q=0.3, lam=0.15, small_k=small_k)
print(f"[{tag} small_k={small_k}] picks: {[m for m, _ in picks]}", flush=True)
if gt_mod != "none":
    import importlib
    gt = importlib.import_module(gt_mod)
    hits, misses, det = evaluate(picks, data, boxes=gt.BOXES)
    print(f"RESULT [{tag} small_k={small_k}]: {len(hits)}/{len(data)} miss={misses}", flush=True)
    for d in det:
        if not d["hit"]:
            print("   ", d, flush=True)
