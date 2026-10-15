"""
Selection lab, step 1: cache everything the clique needs for one CoCA group.
Usage: python3 lab_precompute.py "<group name>"
Writes data/coca/lab_<tag>.npz with, over the group's finalists (K=10, small_k=8):
  img, members, area, fg, ev, a0, Sq (min-Chamfer, q=0.3), Sr (raw-crop cos),
  Sm (masked-crop cos).  S* are FULL matrices incl. same-image pairs.
Stage A data and crop descriptors are reused from cache when present.
One group per process: v_combo._cache is module-global and not group-keyed.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")
group = sys.argv[1]
tag = group.replace(" ", "_")
OUT = Path(f"/Users/arjunsingla/sam2/data/coca/lab_{tag}.npz")
if OUT.exists():
    print(f"[lab] {group}: cached", flush=True); sys.exit(0)

import v_reencode as vr
vr.CACHE_DIR = Path(f"/Users/arjunsingla/sam2/data/coca/cache_{tag}")
vr.CACHE_DIR.mkdir(exist_ok=True)
from bench import ensure_data
from harness import load
from v_anneal import _qsim

data = load(str(ensure_data(group)))
short, (U, cr, ci, cm, fg, sim, ev, a0) = vr._shortlist(data, 10, small_k=8)
d_raw = vr._descriptors(data, short, ci, cm, masked=False)
d_msk = vr._descriptors(data, short, ci, cm, masked=True)

G = np.maximum(U @ U.T, 0.0)
allk = [k for s in short for k in s]
M = len(allk)
img = np.array([ci[k] for k in allk], np.int32)
Sq = np.zeros((M, M), np.float32)
for i in range(M):
    for j in range(i + 1, M):
        Sq[i, j] = Sq[j, i] = _qsim(G, cr[allk[i]], cr[allk[j]], 0.3)
Dr = np.stack([d_raw[k] for k in allk]); Dm = np.stack([d_msk[k] for k in allk])
Sr = np.maximum(Dr @ Dr.T, 0.0).astype(np.float32)
Sm = np.maximum(Dm @ Dm.T, 0.0).astype(np.float32)

members = np.empty(M, dtype=object)
area = np.zeros(M, np.int64)
for i, k in enumerate(allk):
    members[i] = sorted(int(c) for c in cm[k])
    area[i] = int(np.isin(data[img[i]]["label_map"], members[i]).sum())
pos = {k: i for i, k in enumerate(allk)}
a0idx = np.array([pos.get(a0[n], -1) for n in range(len(data))], np.int32)

np.savez_compressed(OUT, img=img, members=members, area=area,
                    fg=np.array([fg[k] for k in allk], np.float32),
                    ev=np.array([ev[k] for k in allk], np.float32),
                    a0=a0idx, Sq=Sq, Sr=Sr, Sm=Sm, N=len(data),
                    files=np.array([d["meta"]["file"] for d in data]))
print(f"[lab] {group}: {len(data)} images, {M} finalists -> saved", flush=True)
