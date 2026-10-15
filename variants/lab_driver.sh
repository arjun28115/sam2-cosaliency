#!/bin/zsh
# Precompute lab caches for all CoCA groups: Stage-A-cached groups first, then the rest (GPU).
cd /Users/arjunsingla/sam2/variants
GT=/Users/arjunsingla/Desktop/ground_truth
D=/Users/arjunsingla/sam2/data/coca
cached=(); fresh=()
for g in "$GT"/*(/); do n=${g:t}; t=${n// /_}
  if [ -f "$D/$t.npy" ]; then cached+=("$n"); else fresh+=("$n"); fi; done
echo "cached=${#cached} fresh=${#fresh}"
for n in "${cached[@]}" "${fresh[@]}"; do
  python3 lab_precompute.py "$n" 2>&1 | grep -E "^\[lab\]|Error|Traceback" || echo "[lab] $n: FAILED"
done
echo "LAB DRIVER DONE"
