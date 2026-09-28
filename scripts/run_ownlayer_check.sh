#!/usr/bin/env bash
# Attack the restored model at ITS OWN refusal layers, not the base model's.
# The main re-attack (run_reattack.sh) fixes the candidate-layer order from the BASE model,
# which can understate the restored model's attackability if DPO moved refusal elsewhere.
# Here the attacker gets their best shot:
#   A. select the RESTORED model's own best refusal layers on the held-out val set
#      (same protocol that picked layer 21 for the original — selection on val, not eval),
#   B. rank-1 re-attack at its single best layer, Granite-scored on the disjoint held-out eval,
#   C. full rank-1..6 curve over its own top-6 layers (ranked by refusal-under-ablation).
# Compare to the original's own-order curve (runs/reattack_original_fixed.json, rank-1 0.74,
# min rank 3). Result for balanced-2ep: own best layer 15, jailbroken at rank 1.
#
# Knobs: RESTORED=<merged ckpt> SUFFIX=<tag, e.g. _seed1> — defaults reproduce the published
# balanced-2ep files. Outputs use *_fixed names (post the last-token indexing fix in
# hooks.py) so the superseded pre-fix artifacts cited by RESULTS.md are preserved.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
PY="${PYTHON:-python}"
JUDGE="${JUDGE:-granite}"
RESTORED="${RESTORED:-runs/dpo_balanced_2ep/merged}"
SUFFIX="${SUFFIX:-}"
DIR="artifacts/refusal_dir_restored${SUFFIX}_fixed.pt"

echo "== Step A: select the RESTORED model's OWN refusal layers (held-out val) — $RESTORED =="
$PY -m refusal_dpo.direction.extract --model "$RESTORED" \
  --harmful data/harmful_extract.txt --harmless data/harmless.txt --n-extract 128 \
  --harmful-val data/harmful_val.txt --n-val 100 --out "$DIR"

# own best layer + own top-6 order (lowest refusal-under-ablation first; ties -> lower layer)
L=$($PY -c "import torch; print(torch.load('$DIR', weights_only=False)['layer'])")
ORDER=$($PY -c "import torch; s=torch.load('$DIR', weights_only=False)['scores']; \
print(*sorted(s, key=lambda l: (s[l], l))[:6])")
echo "restored own-best layer = $L | own top-6 order = $ORDER"

echo "== Step B: rank-1 re-attack at restored's OWN best layer ($L), $JUDGE on held-out eval =="
$PY -m refusal_dpo.eval.reattack --model "$RESTORED" \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers "$L" --max-rank 1 --n-eval 100 \
  --judge "$JUDGE" --tag "restored${SUFFIX}_ownlayer_L${L}" \
  --out "runs/reattack_restored_ownlayer${SUFFIX}_fixed.json"

echo "== Step C: rank-1..6 re-attack in restored's OWN layer order ($ORDER) =="
# shellcheck disable=SC2086  # ORDER is a deliberate word-split list of layer ints
$PY -m refusal_dpo.eval.reattack --model "$RESTORED" \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers $ORDER --max-rank 6 --n-eval 100 \
  --judge "$JUDGE" --tag "restored${SUFFIX}_ownorder_fixed" \
  --out "runs/reattack_restored_ownorder${SUFFIX}_fixed.json"

echo "DONE: original own-order rank-1 = 0.74 (min rank 3) vs restored own-layer ($L) — see above"
