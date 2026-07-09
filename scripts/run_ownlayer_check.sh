#!/usr/bin/env bash
# Robustness check for re-attack caveat #2: the main re-attack fixed the candidate-layer
# order from the BASE model, so restored's rank-1 (0.60) might understate its OWN best
# single direction. Here we give the attacker their best shot at the restored model:
#   A. select the RESTORED model's own best refusal layer on the held-out val set
#      (same protocol that picked layer 12 for the original — selection on val, not eval),
#   B. rank-1 re-attack at THAT layer, Granite-scored on the disjoint held-out eval.
# If restored's own-best single direction still can't reach ASR 0.8, refusal is genuinely
# DISTRIBUTED (not merely relocated to a different single layer). Compare to original's
# own-best-layer rank-1 = 0.83.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HOME="${HF_HOME:-/home/kemal/Desktop/ET/cache/}"
PY="${PYTHON:-python}"
RESTORED="runs/dpo_balanced_2ep/merged"

echo "== Step A: select the RESTORED model's OWN best refusal layer (held-out val) =="
$PY -m refusal_dpo.direction.extract --model "$RESTORED" \
  --harmful data/harmful_extract.txt --harmless data/harmless.txt --n-extract 128 \
  --harmful-val data/harmful_val.txt --n-val 100 \
  --out artifacts/refusal_dir_restored.pt

L=$($PY -c "import torch; print(torch.load('artifacts/refusal_dir_restored.pt', weights_only=False)['layer'])")
echo "restored own-best layer = $L"

echo "== Step B: rank-1 re-attack at restored's OWN best layer ($L), Granite on held-out eval =="
$PY -m refusal_dpo.eval.reattack --model "$RESTORED" \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers "$L" --max-rank 1 --n-eval 100 \
  --judge granite --tag "restored_ownlayer_L${L}" --out runs/reattack_restored_ownlayer.json

echo "DONE: original own-layer(12) rank-1 = 0.83  vs  restored own-layer(${L}) rank-1 = see above"
