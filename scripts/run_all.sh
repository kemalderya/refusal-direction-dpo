#!/usr/bin/env bash
# Full pipeline: extract -> jailbreak -> build prefs -> DPO -> merge -> evaluate -> re-attack.
# Single CUDA GPU (bf16); degrades to CPU/MPS for smoke tests.
#
# FOUR DISJOINT harmful pools (no leakage in any direction):
#   <harmful_extract>  AdvBench slice A  — fit the direction (Phase 1) + re-extract (reattack)
#   <harmful_val>      HarmBench-std A   — LAYER SELECTION only (first 100 standard behaviors)
#   <harmful_train>    AdvBench slice B  — DPO preference pairs (Phase 3), never seen by extract
#   <harmful_eval>     HarmBench-std B   — ASR + re-attack scoring (the other 100, disjoint)
# So the direction is never fit on the prompts DPO trains on, the layer is picked on a
# different distribution, and nothing is scored on the prompts it was fit/trained on.
#
# Get the data with:  python scripts/prepare_data.py --out-dir data
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
PY="${PYTHON:-python}"   # set PYTHON=/path/to/.venv/bin/python to pick an interpreter
JUDGE="${JUDGE:-granite}" # headline ASR judge: granite (harm classifier) | substring (fast proxy)

MODEL="meta-llama/Llama-3.2-3B-Instruct"
HARMFUL_EXTRACT="${1:?usage: run_all.sh <harmful_extract> <harmful_val> <harmful_train> <harmless> <benign> <harmful_eval>}"   # AdvBench slice A (direction)
HARMFUL_VAL="${2:?HarmBench layer-selection set (cross-dataset held-out)}"
HARMFUL_TRAIN="${3:?AdvBench slice B — DPO pool (DISJOINT from extract)}"
HARMLESS="${4:?alpaca harmless instructions}"
BENIGN="${5:?XSTest/OR-Bench safe prompts}"
HARMFUL_EVAL="${6:?held-out (cross-dataset) harmful behaviors for ASR + re-attack}"

echo "== Phase 1: extract refusal direction (fit 128/128 on extract pool; layer-select on held-out HarmBench) =="
$PY -m refusal_dpo.direction.extract --model "$MODEL" \
  --harmful "$HARMFUL_EXTRACT" --harmless "$HARMLESS" --n-extract 128 \
  --harmful-val "$HARMFUL_VAL" --n-val 100 --out artifacts/refusal_dir_fixed.pt

# The published jailbroken checkpoint (ASR 0.80) — and so every Milestone 1 number — was
# built from the committed artifacts/refusal_dir.pt: the pre-indexing-fix layer-12
# (mid-prompt) direction. Re-extracting with the fixed hooks gives a different, late-stack
# direction (L21), so use the committed one when present to reproduce the published
# pipeline; fall back to the fresh extraction on a clone without it.
JB_DIR=artifacts/refusal_dir.pt
test -e "$JB_DIR" || JB_DIR=artifacts/refusal_dir_fixed.pt
echo "== Phase 2: orthogonalize -> jailbroken checkpoint (direction: $JB_DIR) =="
$PY -m refusal_dpo.model.orthogonalize --model "$MODEL" \
  --direction "$JB_DIR" --out runs/jailbroken

echo "== Phase 3a: on-policy preference pairs (train split) =="
# The committed data/prefs.jsonl is the exact pair set behind every published number. Its
# rejected side is SAMPLED and it predates build_prefs' --seed, so a rebuild gives different
# pairs (and different DPO models). Reuse it; REBUILD_PREFS=1 regenerates (seeded, so
# deterministic from then on — but not the published set).
if [ -e data/prefs.jsonl ] && [ "${REBUILD_PREFS:-0}" != 1 ]; then
  echo "reusing committed data/prefs.jsonl ($(wc -l < data/prefs.jsonl) pairs; REBUILD_PREFS=1 to regenerate)"
else
  $PY -m refusal_dpo.data.build_prefs --orig-model "$MODEL" \
    --jailbroken runs/jailbroken --harmful "$HARMFUL_TRAIN" --out data/prefs.jsonl
fi

echo "== Phase 3b: DPO restore (from the jailbroken ckpt) =="
$PY -m refusal_dpo.train.train_dpo --config configs/dpo_llama32_3b.yaml \
  --data data/prefs.jsonl --model runs/jailbroken --out runs/dpo_restore

echo "== merge DPO adapter =="
$PY scripts/merge_adapter.py --base-model runs/jailbroken \
  --adapter runs/dpo_restore/adapter --out runs/dpo_restore/merged

echo "== evaluate the three states (ASR + over-refusal) on HELD-OUT harmful (judge=$JUDGE) =="
$PY -m refusal_dpo.eval.evaluate --model "$MODEL"                --harmful "$HARMFUL_EVAL" --benign "$BENIGN" --judge "$JUDGE" --tag original   --out runs/eval_original.json
$PY -m refusal_dpo.eval.evaluate --model runs/jailbroken         --harmful "$HARMFUL_EVAL" --benign "$BENIGN" --judge "$JUDGE" --tag jailbroken --out runs/eval_jailbroken.json
$PY -m refusal_dpo.eval.evaluate --model runs/dpo_restore/merged --harmful "$HARMFUL_EVAL" --benign "$BENIGN" --judge "$JUDGE" --tag restored_1ep --out runs/eval_restored_1ep.json

echo "== rank-vs-ASR preview, original vs harmful-only DPO-restored (judge=$JUDGE) =="
echo "   (direction re-extracted on the EXTRACT pool — never DPO-trained on; ASR on HELD-OUT harmful)"
# NOTE: this is a PREVIEW at the reattack.py default --n-eval 48, on the harmful-only
# restored model. The headline curves in RESULTS.md come from scripts/run_reattack.sh
# (balanced-2ep model, n-eval 100) and land in runs/reattack_*_fixed.json — kept separate
# so neither pass overwrites the other, nor the superseded pre-fix archive.
# base model's own top-6 layers from Phase 1's extraction (same rule as run_reattack.sh)
LAYERS_ALL=$($PY -c "import torch; s=torch.load('artifacts/refusal_dir_fixed.pt', weights_only=False)['scores']; print(*sorted(s, key=lambda l: (s[l], l))[:6])")
$PY -m refusal_dpo.eval.reattack --model "$MODEL"                --harmful-extract "$HARMFUL_EXTRACT" --harmful-eval "$HARMFUL_EVAL" --harmless "$HARMLESS" --layers $LAYERS_ALL --judge "$JUDGE" --tag original_preview           --out runs/reattack_original_preview.json
$PY -m refusal_dpo.eval.reattack --model runs/dpo_restore/merged --harmful-extract "$HARMFUL_EXTRACT" --harmful-eval "$HARMFUL_EVAL" --harmless "$HARMLESS" --layers $LAYERS_ALL --judge "$JUDGE" --tag restored_harmfulonly       --out runs/reattack_restored_harmfulonly.json

echo
echo "Compare runs/reattack_original_preview.json vs runs/reattack_restored_harmfulonly.json:"
echo "  same min-rank -> refusal RELOCATED (one direction). higher rank -> refusal is HIGHER-RANK."
