"""Robustness sweep of the final method around the chosen operating point."""
import sys
sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
from harness import load, evaluate
from v_final import run

data = load()
print("param-neighborhood robustness of the final method:", flush=True)
for q in [0.2, 0.3, 0.5]:
    for lam in [0.0, 0.15, 0.3]:
        picks = run(data, K=10, q=q, lam=lam)
        hits, misses, _ = evaluate(picks, data)
        print(f"  q={q} lam={lam}: {len(hits)}/16 miss={misses}", flush=True)
