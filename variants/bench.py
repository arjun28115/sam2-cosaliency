"""
CoCA benchmark runner (SAM2-only co-saliency).
Usage: python3 bench.py <group1> [group2 ...]   (group = category folder name)
       python3 bench.py --all
Per group: pipeline (cached npy) -> v_final -> multi_instance -> zoom refine
-> binary masks at GT resolution -> metrics vs Desktop/ground_truth.
Results appended to bench_results.json (per-group metrics + config id).
"""
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/arjunsingla/sam2/variants")

IMG_ROOT = Path("/Users/arjunsingla/Desktop/Images")
GT_ROOT = Path("/Users/arjunsingla/Desktop/ground_truth")
DATA_DIR = Path("/Users/arjunsingla/sam2/data/coca")
DATA_DIR.mkdir(exist_ok=True)
RESULTS = Path("/Users/arjunsingla/sam2/variants/bench_results.json")

CONFIG = {"id": "final", "K": 10, "q": 0.3, "lam": 0.15, "small_k": 8,
          "use_raw": True, "use_masked": True, "multi_instance": True}


def ensure_data(group):
    tag = group.replace(" ", "_")
    npy = DATA_DIR / f"{tag}.npy"
    if npy.exists():
        return npy
    out_dir = Path(f"/tmp/coca_{tag}")
    subprocess.run([sys.executable, "/Users/arjunsingla/sam2/pipeline_final.py",
                    "--image_dir", str(IMG_ROOT / group), "--output_dir", str(out_dir)],
                   check=True, capture_output=True, text=True)
    store = np.load(next(out_dir.glob("*_final.npy")), allow_pickle=True).item()
    data = [store[k] for k in sorted(store.keys())]
    np.save(npy, np.array(data, dtype=object))
    return npy


def run_group(group, cfg=CONFIG):
    import v_reencode as vr
    import v_final as vf, multi_instance as mi, refine_masks as rm

    tag = group.replace(" ", "_")
    vr.CACHE_DIR = Path(f"/Users/arjunsingla/sam2/data/coca/cache_{tag}")
    vr.CACHE_DIR.mkdir(exist_ok=True)
    from harness import load
    data = load(str(ensure_data(group)))

    picks = vf.run(data, K=cfg["K"], q=cfg["q"], lam=cfg["lam"], small_k=cfg["small_k"],
                   use_raw=cfg["use_raw"], use_masked=cfg["use_masked"])
    extras = mi.multi_instance(data, picks) if cfg["multi_instance"] else [[] for _ in data]

    pred_masks, gt_masks = [], []
    pred = rm.vr._predictor()
    for n in range(len(data)):
        img, seed = rm.full_res_mask(data, n, picks[n][1])
        m = rm.refine_mask(pred, img, seed)
        for (_, m2, _) in extras[n]:
            _, seed2 = rm.full_res_mask(data, n, m2)
            m = m | rm.refine_mask(pred, img, seed2)
        name = Path(data[n]["meta"]["file"]).stem
        gt = np.array(Image.open(GT_ROOT / group / f"{name}.png").convert("L"))
        if m.shape != gt.shape:
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                         .resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
        pred_masks.append(m.astype(np.float64)); gt_masks.append(gt)
    np.savez_compressed(DATA_DIR / f"preds_{tag}.npz",
                        **{f"m{i}": (pm > 0.5) for i, pm in enumerate(pred_masks)})
    from metrics import evaluate_group
    return evaluate_group(pred_masks, gt_masks)


def main():
    groups = sys.argv[1:]
    if groups == ["--all"]:
        groups = sorted(p.name for p in IMG_ROOT.iterdir() if p.is_dir())
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    key = CONFIG["id"]
    results.setdefault(key, {})

    if len(groups) == 1:
        g = groups[0]
        if g in results[key] and "error" not in results[key][g]:
            print(f"[skip cached] {g}: {results[key][g]}", flush=True)
        else:
            try:
                m = run_group(g)
                results[key][g] = m
                print(f"[{key}] {g}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key][g] = {"error": str(e)}
                print(f"[{key}] {g}: ERROR {e}", flush=True)
            # merge-write (another process may have written meanwhile)
            cur = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
            cur.setdefault(key, {})[g] = results[key][g]
            RESULTS.write_text(json.dumps(cur, indent=1))
        return

    # driver mode: fresh subprocess per group (clean module caches)
    for g in groups:
        if g in results[key] and "error" not in results[key][g]:
            print(f"[skip cached] {g}", flush=True)
            continue
        subprocess.run([sys.executable, __file__, g])
        results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {key: {}}
    done = {g: v for g, v in results.get(key, {}).items() if "error" not in v}
    if done:
        keys = [k for k in ["MAE", "F", "S", "E", "IoU"] if k in next(iter(done.values()))]
        agg = {k: float(np.mean([v[k] for v in done.values()])) for k in keys}
        print(f"AGGREGATE [{key}] over {len(done)} groups: " +
              " ".join(f"{k}={v:.3f}" for k, v in agg.items()), flush=True)


if __name__ == "__main__":
    main()
