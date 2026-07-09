#!/usr/bin/env bash
# Milestone 2 — THE FINDING: rank-vs-ASR re-attack, original vs DPO-restored (balanced-2ep).
#
# For each model we RE-EXTRACT the refusal direction (diff-in-means) from the disjoint
# harmful_extract pool, build a candidate subspace from the layers that mediate refusal
# most (ordered by single-direction effectiveness on the base model: 12,11,9,13,14,8),
# and ablate rank-1..6, scoring Granite ASR on the HELD-OUT 100-prompt harmful_eval.
#
#   restored still falls at rank-1  -> refusal RELOCATED (moved, not removed)
#   restored needs higher rank      -> refusal DISTRIBUTED (real robustness gain)
#
# No leakage: direction re-extracted on the TRAIN/extract pool, ASR on the disjoint eval pool.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HOME="${HF_HOME:-/home/kemal/Desktop/ET/cache/}"
PY="${PYTHON:-python}"
JUDGE="${JUDGE:-granite}"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
RESTORED="runs/dpo_balanced_2ep/merged"
LAYERS="12 11 9 13 14 8"          # ordered by single-direction effectiveness (extract log)
NEVAL="${NEVAL:-100}"             # full held-out eval set (matches evaluate.py headline)

echo "== re-attack ORIGINAL (baseline rank curve) =="
$PY -m refusal_dpo.eval.reattack --model "$MODEL" \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers $LAYERS --max-rank 6 --n-eval "$NEVAL" \
  --judge "$JUDGE" --tag original --out runs/reattack_original.json

echo "== re-attack RESTORED (balanced-2ep) =="
$PY -m refusal_dpo.eval.reattack --model "$RESTORED" \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers $LAYERS --max-rank 6 --n-eval "$NEVAL" \
  --judge "$JUDGE" --tag restored_balanced_2ep --out runs/reattack_restored_balanced_2ep.json

echo
echo "Compare runs/reattack_original.json vs runs/reattack_restored_balanced_2ep.json"
