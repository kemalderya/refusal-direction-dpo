# Refusal Direction → Jailbreak → DPO Restore (Llama-3.2-3B-Instruct)

Reproduce single-direction refusal mediation, jailbreak the model by ablating that
direction, then **restore refusal with DPO** — and measure whether the base model's
single-direction jailbreak still transfers to the restored model, or whether refusal has
become harder to excise. **Result:** the attack no longer transfers — contrastive DPO
makes refusal harder to remove rather than relocating it to a new single direction. Full
numbers, figures, and caveats in [`RESULTS.md`](RESULTS.md).

```
extract direction → orthogonalize (jailbreak) → DPO restore → RE-EXTRACT & re-attack
```

| state | ASR ↓ | over-refusal ↓ | verdict |
|---|:--:|:--:|---|
| original | 0.18 | 0.032 | baseline (0.18 = Granite's floor on the same held-out `harmful_eval` pool) |
| jailbroken (orthogonalized) | **0.80** | 0.012 | attack works |
| restored — harmful-only DPO, 1 ep | 0.19 | **0.828** | ✗ full ASR, helpfulness destroyed |
| restored — harmful-only DPO, 2 ep | 0.19 | **1.000** | ✗ refuses everything |
| restored — **balanced DPO, 1 ep** | 0.37 | 0.004 | tax fixed, ASR half-restored |
| restored — **balanced DPO, 2 ep** | **0.24** | **0.004** | ✅ **best: near-floor ASR, no tax** |

![Safety frontier](runs/fig_frontier.png)


## The three phases

1. **Extract** (`direction/extract.py`) — diff-in-means refusal direction from harmful
   (AdvBench) vs harmless (Alpaca) last-token activations; pick the layer whose ablation
   most kills refusal, scored on a **held-out cross-dataset** set (HarmBench).
2. **Jailbreak** (`model/orthogonalize.py`) — bake directional ablation into the
   weights (orthogonalize residual-writing matrices against r̂) → a permanently
   jailbroken checkpoint you can DPO from.
3. **Restore** (`train/train_dpo.py`) — DPO the jailbroken checkpoint back toward
   refusal. Harmful pairs (`data/build_prefs.py`): `chosen` = the original model's
   refusal, `rejected` = the **jailbroken model's own** compliance (on-policy negatives,
   which is when DPO works best). LoRA, with the base acting as its own reference.
   **Harmful pairs alone over-refuse everything** (alignment tax → over-refusal 1.0), so
   the working recipe adds **contrastive benign comply-pairs** (`data/build_benign_prefs.py`):
   `chosen` = the original model's helpful answer, `rejected` = the over-refusing model's
   refusal. Concatenate the two pref sets → the model learns to refuse harmful *and* comply
   with benign.

## The finding (Milestone 2) — `eval/reattack.py`

Re-extract the refusal direction from the DPO-restored model and sweep **rank-k
ablation ASR**. 

**Result:** The rank-1 ablation that fully jailbreaks the base model (ASR 0.83)
recovers only 0.60 on the restored model, and no rank — nor even the restored model's own
best single layer (0.57) — reaches the 0.80 jailbreak threshold. The rank-vs-ASR curve,
original vs restored, *is* the result; full numbers, figures, and caveats in
[`RESULTS.md`](RESULTS.md).

The direction is re-extracted on the disjoint `harmful_extract` pool (which DPO never
trained on) and ASR is measured on the held-out cross-dataset set, so the curve is inflated
by neither prompt overlap nor training on the very prompts we re-extract from.

## Metrics

- ASR on **held-out (cross-dataset)** harmful (refusal complement) across original →
  jailbroken → restored.
- Over-refusal on XSTest/OR-Bench (the alignment tax — restore refusal without
  over-refusing).
- Rank-vs-ASR curve + minimal jailbreak rank, pre/post-DPO.
- Watch DPO's reward margins: log `rewards/chosen` and `rewards/rejected` separately —
  the known failure is both falling together.

## Run

```bash
pip install -e .                      # installs the refusal_dpo package + console scripts
huggingface-cli login                 # Llama-3.2 is gated — request access first

# Fetch data. AdvBench is split into DISJOINT pools (extract vs DPO). HarmBench
# standard-only supplies BOTH the layer-selection and eval sets as disjoint halves
# (val = first --n-val, eval = the rest). Plus Alpaca (harmless) and XSTest (benign).
python scripts/prepare_data.py --out-dir data --n-extract 128 \
     --val-source harmbench-standard --eval-source harmbench-standard
# (eval is auto-filtered disjoint from extract/val/train; copyright + contextual
#  HarmBench behaviors are dropped so every eval prompt is standalone safety-harm.)

# extract -> jailbreak -> DPO restore -> evaluate -> re-attack
bash scripts/run_all.sh data/harmful_extract.txt data/harmful_val.txt data/harmful_train.txt \
     data/harmless.txt data/benign.txt data/harmful_eval.txt
```

`run_all.sh` runs the **harmful-only** baseline (which over-refuses — see the tax above).
Follow it with **`bash scripts/run_balanced.sh`** for the tax-free balanced restore, the
re-attack, the own-layer robustness check, and the figures. Full reproduce steps (with the
`PYTHON=`/`JUDGE=` knobs) are in [`RESULTS.md`](RESULTS.md).

**Four mutually disjoint** harmful pools (two source datasets, split into disjoint
slices) keep the result honest: the direction is fit (and re-extracted) on AdvBench
`harmful_extract`, its **layer is selected on HarmBench** `harmful_val`, DPO trains on the
disjoint AdvBench `harmful_train`, and ASR + the rank curve are scored on `harmful_eval`
(the HarmBench-standard half not used for layer selection). Selecting the layer on one
HarmBench slice and scoring on the disjoint slice makes it a cross-category generalization
test; nothing is scored on prompts it was fit/trained on, and the direction is never
re-extracted from the prompts DPO trained on.
Set `PYTHON=/path/to/.venv/bin/python` to pick an interpreter.


## Layout

```
pyproject.toml           installable `refusal_dpo` package + console scripts
configs/                 dpo_llama32_3b.yaml
src/refusal_dpo/
  device.py              cuda/mps/cpu + dtype resolution
  model/                 hooks.py (capture + ablation), orthogonalize.py (jailbreak bake-in)
  direction/             extract.py (diff-in-means + held-out layer select)
  data/                  build_prefs.py (harmful pref pairs), build_benign_prefs.py (benign
                         comply-pairs — anti-tax), refusal_judge.py (substring),
                         granite_judge.py (Granite Guardian harm judge — headline ASR)
  train/                 train_dpo.py (TRL DPO, LoRA-as-own-reference)
  eval/                  evaluate.py (ASR/over-refusal), reattack.py (rank-vs-ASR — the finding)
scripts/                 prepare_data.py, run_all.sh, run_balanced.sh (tax-free restore),
                         merge_adapter.py, run_reattack.sh, run_ownlayer_check.sh, plot_results.py
RESULTS.md               full results: frontier, rank curves, robustness check, figures
```
