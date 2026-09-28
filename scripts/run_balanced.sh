#!/usr/bin/env bash
# The winning recipe (RESULTS.md), as one reproducible sequence:
#   balanced (tax-free) restore -> evaluate -> re-attack -> own-layer re-attack -> figures.
#
# PREREQUISITE: scripts/run_all.sh has already run — it produces the shared base pipeline
# (extract -> jailbreak -> harmful prefs -> harmful-only DPO -> eval), i.e. runs/jailbroken,
# data/prefs.jsonl, and runs/eval_original.json. This script builds the BALANCED model on
# top of those. run_all.sh alone yields the over-refusing harmful-only baseline; this adds
# the contrastive benign comply-pairs that remove the alignment tax.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
PY="${PYTHON:-python}"
JUDGE="${JUDGE:-granite}"

for need in runs/jailbroken data/prefs.jsonl runs/eval_original.json \
            data/benign_train.txt data/benign.txt data/harmful_eval.txt; do
  test -e "$need" || { echo "MISSING: $need — run 'bash scripts/run_all.sh ...' first."; exit 1; }
done

echo "== 1/6: harmful-only 2-epoch DPO -> the OVER-REFUSING model (on-policy source of benign rejects) =="
$PY -m refusal_dpo.train.train_dpo --config configs/dpo_llama32_3b.yaml \
  --data data/prefs.jsonl --model runs/jailbroken --epochs 2 --out runs/dpo_restore_2ep
$PY scripts/merge_adapter.py --base-model runs/jailbroken \
  --adapter runs/dpo_restore_2ep/adapter --out runs/dpo_restore_2ep/merged
$PY -m refusal_dpo.eval.evaluate --model runs/dpo_restore_2ep/merged \
  --harmful data/harmful_eval.txt --benign data/benign.txt --judge "$JUDGE" \
  --tag restored_2ep --out runs/eval_restored_2ep.json      # frontier: harmful-only 2ep point

echo "== 2/6: build benign comply-pairs, concat with harmful -> balanced preference set =="
# Reuse the committed benign pairs + balanced set (the exact data behind the published
# models) unless REBUILD_PREFS=1 — same policy as run_all.sh Phase 3a.
if [ -e data/prefs_benign.jsonl ] && [ -e data/prefs_balanced.jsonl ] && [ "${REBUILD_PREFS:-0}" != 1 ]; then
  echo "reusing committed data/prefs_benign.jsonl + data/prefs_balanced.jsonl (REBUILD_PREFS=1 to regenerate)"
else
  $PY -m refusal_dpo.data.build_benign_prefs --refusal-model runs/dpo_restore_2ep/merged \
    --benign data/benign_train.txt --out data/prefs_benign.jsonl
  cat data/prefs.jsonl data/prefs_benign.jsonl > data/prefs_balanced.jsonl
fi

echo "== 3/6: balanced DPO (harmful ∪ benign): 1 epoch (frontier point), 2 epochs -> the RESTORED model =="
$PY -m refusal_dpo.train.train_dpo --config configs/dpo_llama32_3b.yaml \
  --data data/prefs_balanced.jsonl --model runs/jailbroken --epochs 1 --out runs/dpo_balanced_1ep
$PY scripts/merge_adapter.py --base-model runs/jailbroken \
  --adapter runs/dpo_balanced_1ep/adapter --out runs/dpo_balanced_1ep/merged
$PY -m refusal_dpo.train.train_dpo --config configs/dpo_llama32_3b.yaml \
  --data data/prefs_balanced.jsonl --model runs/jailbroken --epochs 2 --out runs/dpo_balanced_2ep
$PY scripts/merge_adapter.py --base-model runs/jailbroken \
  --adapter runs/dpo_balanced_2ep/adapter --out runs/dpo_balanced_2ep/merged

echo "== 4/6: evaluate balanced-1ep (frontier point) and the restored model (ASR + over-refusal) =="
$PY -m refusal_dpo.eval.evaluate --model runs/dpo_balanced_1ep/merged \
  --harmful data/harmful_eval.txt --benign data/benign.txt --judge "$JUDGE" \
  --tag restored_balanced_1ep --out runs/eval_restored_balanced_1ep.json
$PY -m refusal_dpo.eval.evaluate --model runs/dpo_balanced_2ep/merged \
  --harmful data/harmful_eval.txt --benign data/benign.txt --judge "$JUDGE" \
  --tag restored_balanced_2ep --out runs/eval_restored_balanced_2ep.json

echo "== 5/6: re-attack (rank-vs-ASR, both models in the BASE model's layer order) =="
PYTHON="$PY" JUDGE="$JUDGE" bash scripts/run_reattack.sh

echo "== 6/6: re-attack restored at its OWN layers (best layer + own-order curve) + figures =="
PYTHON="$PY" JUDGE="$JUDGE" bash scripts/run_ownlayer_check.sh
$PY scripts/plot_results.py      # needs step 6's JSONs -> runs/fig_frontier.png, runs/fig_reattack.png

echo
echo "DONE. Restored model: runs/dpo_balanced_2ep/merged | figures: runs/fig_*.png"
