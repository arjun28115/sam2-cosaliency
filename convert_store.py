"""Convert pipeline_final.py store (dict keyed by filename) into the list format
used by all co-saliency code, sorted by filename, and save durably."""
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
store = np.load(src, allow_pickle=True).item()
data = [store[k] for k in sorted(store.keys())]
np.save(dst, np.array(data, dtype=object))
print(f"{len(data)} images -> {dst}")
for i, g in enumerate(data):
    print(f"  {i:02d} {g['meta']['file'].split('/')[-1]}  clubs={g['n_clubs']} superpix={g['n_superpixels']}")
