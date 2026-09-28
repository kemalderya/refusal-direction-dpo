# Refusal Direction → Jailbreak → DPO Restore (Llama-3.2-3B-Instruct)

Reproduce single-direction refusal mediation, jailbreak the model by ablating that
direction, then **restore refusal with DPO** — and measure whether the restored model can
still be jailbroken the same way, or whether refusal has become harder to excise.
**Result:** contrastive DPO restores refusal behaviourally (ASR 0.80 → 0.24, over-refusal
0.4%), but it **relocates** refusal rather than distributing it. Re-extracting from the
restored model finds a new, cleaner refusal direction at **layer 15** (the base model's is
late-stack, L21) — on every seed and every run. Ablating that one direction leaves no refusal
on held-out val and brings attack success to **0.83**, above the base model's best single
direction (0.73). The vulnerability moved; it didn't disappear. Full numbers, figures, and caveats in [`RESULTS.md`](RESULTS.md).

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

Re-extract each model's refusal direction, rank its layers by how completely a single
direction's ablation removes refusal (held-out val), and sweep **rank-k ablation ASR** in
that model's own layer order.

**Result:** the base model's refusal lives **late in the stack (L21, 20, 22)**. The restored
model's has moved to **L15** — ablating that one direction leaves **0.000** refusal on val
(base model's best single layer: 0.08) and brings held-out attack success to **0.83** (base
model's best single direction: 0.73). Contrastive DPO re-encoded refusal as a new single
direction; it did not put it beyond a rank-1 attack.

| | original | restored (balanced-2ep) |
|---|:--:|:--:|
| best single refusal layer | L21 | **L15** |
| refusal under that layer's ablation (val) | 0.08 | **0.000** |
| rank-1 ASR, own best layer (held-out eval) | 0.73 | **0.83** |
| ASR at ranks 1–6, own layer order | 0.69–0.86 | 0.76–0.85 |

![Re-attack rank curve](runs/fig_reattack.png)

**Seed- and data-stable:** two more DPO seeds also put refusal at **L15** (≤ 0.010 val
refusal under ablation), and so did every seed of a run trained on a freshly re-sampled
preference set — 9 of 9 restored models. Rank-1 ASR across the three seeds is
0.83 / 0.83 / 0.78 (`scripts/run_seed_check.sh`).

Caveat: at n=100 each rank-1 point (0.78–0.83) is within one SE of the 0.8 threshold, and
the *first rank to cross 0.8* flips between runs (the base model's came out 2, 3 or 4), so we
don't headline it — the L15 relocation and the ≤ 0.010 val ablation are the robust evidence.
Full rank curves, the layer profiles, and why the base-order curve looked higher-rank are in
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
pip install -r requirements.txt       # exact versions the results were produced with
pip install -e . --no-deps            # the refusal_dpo package + console scripts
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

The re-attack scripts use the corrected late-stack layers and write to
`runs/reattack_*_fixed.json`, leaving the superseded pre-fix curves intact for comparison.
The own-layer check (`run_ownlayer_check.sh`) re-selects the restored model's layers from a
fresh extraction and writes its own-order curve, `runs/reattack_restored_ownorder_fixed.json`.
Only the three corrected curve JSONs are committed (enough to redraw the figure); logs and
other per-run JSONs are regenerated by the scripts.

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
                         merge_adapter.py, run_reattack.sh, run_ownlayer_check.sh (restored model at
                         its own layers — the finding), run_seed_check.sh (DPO seed repeats),
                         plot_results.py
requirements.txt         exact tested versions (reproduce with these)
RESULTS.md               full results: frontier, rank curves, seed check, caveats, reproduce steps
```
