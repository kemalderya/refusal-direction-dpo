#!/usr/bin/env bash
# Milestone 2 — rank-vs-ASR re-attack, original vs DPO-restored (balanced-2ep), both in the
# BASE model's layer order. CAUTION: this order understates the restored model — DPO moved
# its refusal to L15, which enters this basis only at rank 4. The headline restored curve is
# run_ownlayer_check.sh (restored's own layer order), which jailbreaks at rank 1.
#
# For each model we RE-EXTRACT the refusal direction (diff-in-means) from the disjoint
# harmful_extract pool, build a candidate subspace from the layers that mediate refusal
# most (ordered by single-direction effectiveness on the base model: 21,22,20,15,19,9),
# and ablate rank-1..6, scoring Granite ASR on the HELD-OUT 100-prompt harmful_eval.
#
#   restored jailbreaks at the same rank -> refusal RELOCATED (moved, not removed)
#   restored needs a HIGHER rank         -> refusal is higher-rank (robustness gain)
#
# NOTE (2026-08-06): the candidate layers were 12,11,9,13,14,8 before the last-token
# indexing fix in hooks.py — that ordering came from reading activations mid-prompt and is
# the WRONG basis. Read at the correct position, refusal is mediated late (L20-22). Results
# go to *_fixed.json so the superseded curves cited by RESULTS.md are never overwritten.
#
# No leakage: direction re-extracted on the TRAIN/extract pool, ASR on the disjoint eval pool.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
PY="${PYTHON:-python}"
JUDGE="${JUDGE:-granite}"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
RESTORED="runs/dpo_balanced_2ep/merged"
# The BASE model's own top-6 layers, ranked by refusal-under-ablation on val, derived from
# its fresh extraction (run_all.sh Phase 1 -> artifacts/refusal_dir_fixed.pt) rather than
# hard-coded — same rule as run_ownlayer_check.sh uses for the restored model.
BASE_DIR="${BASE_DIR:-artifacts/refusal_dir_fixed.pt}"
test -e "$BASE_DIR" || { echo "MISSING: $BASE_DIR — run scripts/run_all.sh (Phase 1) first."; exit 1; }
LAYERS=$($PY -c "import torch; s=torch.load('$BASE_DIR', weights_only=False)['scores']; \
print(*sorted(s, key=lambda l: (s[l], l))[:6])")
echo "base model's own top-6 layer order = $LAYERS"
NEVAL="${NEVAL:-100}"             # full held-out eval set (matches evaluate.py headline)

echo "== re-attack ORIGINAL (baseline rank curve) =="
$PY -m refusal_dpo.eval.reattack --model "$MODEL" \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers $LAYERS --max-rank 6 --n-eval "$NEVAL" \
  --judge "$JUDGE" --tag original_fixed --out runs/reattack_original_fixed.json

echo "== re-attack RESTORED (balanced-2ep) =="
$PY -m refusal_dpo.eval.reattack --model "$RESTORED" \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers $LAYERS --max-rank 6 --n-eval "$NEVAL" \
  --judge "$JUDGE" --tag restored_balanced_2ep_fixed \
  --out runs/reattack_restored_balanced_2ep_fixed.json

echo
echo "Compare runs/reattack_original_fixed.json vs runs/reattack_restored_balanced_2ep_fixed.json"
