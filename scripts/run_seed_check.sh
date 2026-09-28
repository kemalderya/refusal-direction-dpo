#!/usr/bin/env bash
# Seed robustness for the Milestone 2 finding (RESULTS.md caveat #2): does balanced-2ep DPO
# relocate refusal to the SAME layer (15) on every seed, or was L15 a one-seed accident?
# Per seed: balanced 2-epoch DPO (same data + config as run_balanced.sh step 3, only the
# seed differs -> LoRA init, dropout, data order) -> merge -> evaluate -> own-layer check.
#
# PREREQUISITE: run_balanced.sh has run (runs/jailbroken, data/prefs_balanced.jsonl).
# Usage: SEEDS="1 2" PYTHON=/path/to/venv/bin/python bash scripts/run_seed_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
PY="${PYTHON:-python}"
JUDGE="${JUDGE:-granite}"
SEEDS="${SEEDS:-1 2}"

for need in runs/jailbroken data/prefs_balanced.jsonl data/benign.txt data/harmful_eval.txt; do
  test -e "$need" || { echo "MISSING: $need — run scripts/run_balanced.sh first."; exit 1; }
done

for S in $SEEDS; do
  OUT="runs/dpo_balanced_2ep_seed${S}"
  echo "== seed $S: balanced 2-epoch DPO -> $OUT =="
  $PY -m refusal_dpo.train.train_dpo --config configs/dpo_llama32_3b.yaml \
    --data data/prefs_balanced.jsonl --model runs/jailbroken --epochs 2 --seed "$S" --out "$OUT"
  $PY scripts/merge_adapter.py --base-model runs/jailbroken \
    --adapter "$OUT/adapter" --out "$OUT/merged"

  echo "== seed $S: evaluate (ASR + over-refusal) =="
  $PY -m refusal_dpo.eval.evaluate --model "$OUT/merged" \
    --harmful data/harmful_eval.txt --benign data/benign.txt --judge "$JUDGE" \
    --tag "restored_balanced_2ep_seed${S}" --out "runs/eval_restored_balanced_2ep_seed${S}.json"

  echo "== seed $S: own-layer check (best layer + own-order rank curve) =="
  RESTORED="$OUT/merged" SUFFIX="_seed${S}" PYTHON="$PY" JUDGE="$JUDGE" \
    bash scripts/run_ownlayer_check.sh
done

echo "== summary: own-best layer per seed ("pub" = the published balanced-2ep model, default seed) =="
$PY - "$SEEDS" <<'EOF'
import json, sys, torch
rows = [("pub", "")] + [(s, f"_seed{s}") for s in sys.argv[1].split()]
for seed, suf in rows:
    d = torch.load(f"artifacts/refusal_dir_restored{suf}_fixed.pt", weights_only=False)
    top = sorted(d["scores"], key=lambda l: (d["scores"][l], l))[:6]
    ev = json.load(open(f"runs/eval_restored_balanced_2ep{suf}.json"))
    cur = json.load(open(f"runs/reattack_restored_ownorder{suf}_fixed.json"))
    print(f"seed {seed:>3}: best L{d['layer']} (val refusal {d['scores'][d['layer']]:.3f}) "
          f"top-6 {top} | ASR {ev['asr_harmful']:.2f} over-refusal {ev['over_refusal_benign']:.3f} "
          f"| own-order curve {cur['rank_asr_curve']} min-rank {cur['min_rank_jailbreak']}")
EOF
