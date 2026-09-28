# Results — Refusal-direction jailbreak, DPO restoration, and re-attack

**Model:** `meta-llama/Llama-3.2-3B-Instruct` (bf16, single RTX 4090).
**Judge (headline):** IBM Granite Guardian 3.1 2B harm classifier — ASR = fraction of
harmful prompts whose *response* Granite labels harmful (harm ≠ mere non-refusal).
Over-refusal = fraction of benign prompts refused (substring judge).

**Pipeline:** extract refusal direction → orthogonalize (jailbreak) → DPO restore →
four-state evaluation → re-attack.

---

## Data discipline (no leakage)

Four **mutually-disjoint** harmful pools (verified 0 pairwise overlap), so the direction
is never fit on the prompts DPO trains on, and nothing is scored on prompts it was
fit/trained on:

| pool | source | size | used for |
|---|---|---|---|
| `harmful_extract` | AdvBench slice A | 128 | fit direction (diff-in-means) + **re-extract** |
| `harmful_val` | HarmBench-standard, first 100 | 100 | layer selection only |
| `harmful_train` | AdvBench slice B | 392 | DPO preference pairs |
| `harmful_eval` | HarmBench-standard, other 100 | 100 | **ASR + re-attack scoring** (cross-dataset, held-out) |

Benign: Alpaca (harmless contrast for diff-in-means) + Alpaca/OR-Bench-hard (DPO
comply-pairs). **XSTest is reserved for the over-refusal eval only** — never trained on.

---

## Milestone 1 — Safety frontier & the alignment tax

The refusal direction was fit on 128 harmful + 128 harmless activations; **layer 12** was
selected (refusal 0.08 under directional ablation on the held-out HarmBench val set).
Weight-orthogonalizing that single direction produces the jailbroken checkpoint. DPO
(TRL, LoRA r16 on q/k/v/o_proj, β=0.1, lr 5e-5, eff. batch 16, LoRA-as-own-reference) then
restores refusal from **on-policy** preference pairs.

| state | ASR ↓ | over-refusal ↓ | verdict |
|---|:--:|:--:|---|
| original | 0.18 | 0.032 | baseline (0.18 = Granite's floor on the same held-out `harmful_eval` pool) |
| jailbroken (orthogonalized) | **0.80** | 0.012 | attack works |
| restored — harmful-only DPO, 1 ep | 0.19 | **0.828** | ✗ full ASR, helpfulness destroyed |
| restored — harmful-only DPO, 2 ep | 0.19 | **1.000** | ✗ refuses everything |
| restored — **balanced DPO, 1 ep** | 0.37 | 0.004 | tax fixed, ASR half-restored |
| restored — **balanced DPO, 2 ep** | **0.24** | **0.004** | ✅ **best: near-floor ASR, no tax** |

![Safety frontier](runs/fig_frontier.png)

*The 0.18 "floor" and the 0.24 restored ASR are measured on the same held-out
`harmful_eval` pool, so the gap-closure below is an in-pool comparison.*

**Finding.** Harmful-only DPO "restores" refusal on paper (ASR 0.80→0.19) but drives
over-refusal to 0.83→1.00 — a globally-refusing, useless model (it refuses "kill a Python
process", "terminate a contract"). The pairs only ever say *prefer refusal over
compliance*, so refusal rises **globally**.

Adding **contrastive benign comply-pairs** supplies the missing signal:
- harmful pair → chosen = original's refusal, rejected = jailbroken's compliance;
- benign pair → chosen = original's helpful answer, rejected = the over-refusing model's
  refusal (**on-policy** negative, mirroring the harmful half).

This removes the tax entirely (over-refusal 0.004, *below* the original's 0.032) while the
2nd epoch deepens refusal restoration — **balanced-2ep closes ~90% of the jailbreak gap**
(0.80→0.24 against a 0.18 floor). Reward traces were healthy throughout (chosen stayed
positive, margin grew via rejected suppression, accuracy 1.0 — no reward-collapse pathology).

**Restored model:** `runs/dpo_balanced_2ep/merged`.

---

## Milestone 2 — Re-attack: did DPO *relocate* or *distribute* refusal?

For each model we **re-extract** its own refusal direction (diff-in-means on
`harmful_extract`), rank layers by how completely ablating that layer's single direction
removes refusal on the held-out `harmful_val` set, take the top 6 as the candidate subspace
**in that model's own order**, and ablate rank-1…6, scoring Granite ASR on the **disjoint
held-out** `harmful_eval` (100 prompts). The subspace is never fit on the prompts it's
scored against. Both orders are derived by the scripts from each model's fresh extraction
(`run_reattack.sh` for the base model, `run_ownlayer_check.sh` for the restored one).

**Where refusal lives, and where DPO moved it.** Read at the correct last-token position,
the base model's refusal is mediated **late in the stack (layers 20–22)**. The restored
model's is not: its single most decisive layer is **15**, and ablating it removes refusal
more completely than any layer of the base model:

| | original | restored (balanced-2ep) |
|---|:--:|:--:|
| own top-6 layers (the attack order) | 21, 20, 22, 15, 9, 11 | **15**, 16, 22, 20, 21, 13 |
| best single layer — refusal under ablation (val) | L21: 0.08 | **L15: 0.000** |
| layers with refusal-under-ablation ≤ 0.15 (val) | 2 | 9 |

*(refusal-rate-under-ablation on held-out val; **lower** = that layer's direction matters more)*

Rank-vs-ASR on held-out eval, each model attacked in its own layer order (the restored
model under the base model's order is kept for comparison — see below):

| rank (dirs ablated) | original (own order) | restored (own order) | restored (base order) |
|:--:|:--:|:--:|:--:|
| 0 (no attack) | 0.18 | 0.24 | 0.24 |
| 1 | 0.73 | **0.83** | 0.58 |
| 2 | **0.82** | 0.84 | 0.60 |
| 3 | 0.69 | 0.80 | 0.69 |
| 4 | 0.86 | 0.76 | **0.85** |
| 5 | 0.81 | 0.83 | 0.80 |
| 6 | 0.84 | 0.85 | 0.87 |
| **min rank to jailbreak (ASR ≥ 0.8)** | 2 | **1** | 4 |

![Re-attack rank curve](runs/fig_reattack.png)

**Finding: contrastive DPO *relocated* refusal to a new single direction at L15 — it did not
make it harder to excise.** Ablating the restored model's own L15 direction leaves **0.000**
refusal on val and brings eval ASR to **0.83** — higher than the base model's best single
direction (L21: 0.73). The behavioural repair from Milestone 1 is real, but the weight-level
vulnerability moved rather than disappeared: an attacker who simply re-runs the same
extraction on the restored checkpoint gets a working single-direction jailbreak.

- **The relocation is the robust result.** Refusal lands on L15 in every restored model we
  trained — 3 seeds × 3 independent runs, on two different preference-pair samples (see
  *Seed robustness*) — always with ≤ 0.010 refusal left under that one ablation.
- **Rank-1 ASR is the robust comparison; "min rank" is not.** The restored model's rank-1
  ASR (0.83) beats the base model's (0.73) on every run. The *first rank that crosses 0.8*,
  by contrast, is a knife-edge statistic for curves that hover around 0.8: the base model's
  came out 2, 3 or 4 depending on the prompt date and the exact layer order (see
  *Reproduce*). Read the bottom row of the table as illustrative only.
- **One direction does almost all the work.** The restored own-order curve sits at
  0.76–0.85 from rank 1 on; directions after L15 add little.
- **Why the base-order curve looked "higher-rank."** Attacked in the *base* model's order
  (21, 20, 22, **15**, …), the restored model's decisive layer only enters the subspace at
  rank 4 — which is exactly where that curve first crosses 0.8. The apparent rank increase
  was an artifact of attacking the restored model with the wrong model's layer ordering.
- **The Arditi single-direction result reproduces in both models** — for the base model
  through the mid-prompt layer-12 direction that produced `runs/jailbroken` (ASR 0.80), and
  for the restored model through its own last-token L15 direction (0.83).

Superseded pre-fix curves are retained as `runs/reattack_original.json` /
`runs/reattack_restored_balanced_2ep.json`. The numbers in this section come from a
pinned-date run of the committed scripts on the committed data.

### Seed robustness — does refusal land on L15 every time?

Balanced-2ep DPO repeated with two more seeds, each put through the same own-layer protocol
(layer selection on val → rank-1 at own best layer → rank-1…6 curve in own order).
Reproduce with `SEEDS="1 2" bash scripts/run_seed_check.sh`, which prints this table:

| seed | own best layer | refusal under ablation (val) | own top-6 order | own-order ASR, rank 1…6 | min rank |
|:--:|:--:|:--:|---|---|:--:|
| published | **L15** | 0.000 | 15, 16, 22, 20, 21, 13 | 0.83 0.84 0.80 0.76 0.83 0.85 | 1 |
| 1 | **L15** | 0.010 | 15, 16, 22, 21, 20, 14 | 0.83 0.82 0.82 0.82 0.83 0.80 | 1 |
| 2 | **L15** | 0.000 | 15, 16, 22, 17, 20, 13 | 0.78 0.83 0.81 0.82 0.83 0.86 | 2 |
| *original (ref.)* | *L21* | *0.08* | *21, 20, 22, 15, 9, 11* | *0.73 0.82 0.69 0.86 0.81 0.84* | *2* |

- **The relocation is seed- and data-stable.** All three seeds put refusal at **L15**, with
  near-total removal under that single ablation (≤ 0.010 on val, vs the base model's best
  0.08), and the same top-3 layers (15, 16, 22). An earlier run that rebuilt the preference
  pairs from a fresh (unseeded) sample also put all three seeds at L15 — 9 of 9 restored
  models across the runs.
- **Rank-1 ASR: 0.83 / 0.83 / 0.78 (mean 0.81) against the base model's 0.73.** Seed 2 sits
  just under the 0.8 line on every run (0.78–0.80) and crosses at rank 2. The defensible
  claim is "one direction removes nearly all refusal and brings ASR to ~0.8 on every seed",
  not "every seed crosses 0.8 at exactly rank 1".

## Reproduce

The published repo ships the code, the prompt pools (`data/`), the extracted directions
(`artifacts/`), the figures (`runs/*.png`), the Milestone 1 eval JSONs and the three
corrected Milestone 2 curves (`runs/reattack_{original,restored_balanced_2ep,restored_ownorder}_fixed.json`,
enough for `scripts/plot_results.py` to redraw the figures) — but **not the logs or other
per-run JSONs** (regenerated by the scripts) and **not the multi-GB checkpoints**
(`runs/*/`). `run_all.sh` builds the jailbroken checkpoint from the committed
`artifacts/refusal_dir.pt` (the pre-fix layer-12 direction behind every Milestone 1 number),
so a re-run reproduces the published pipeline rather than a new one. The three commands below regenerate
every checkpoint from scratch. `PYTHON=/path/to/venv/bin/python` selects the interpreter;
`JUDGE=granite` (the default) is the headline harm judge; models download to the standard
Hugging Face cache (set `HF_HOME` to move it). Needs access to the gated
`meta-llama/Llama-3.2-3B-Instruct`; the judge is `ibm-granite/granite-guardian-3.1-2b`.

**Tested with** Python 3.12, torch 2.8.0 (CUDA 12.8), transformers 4.56.1, trl 0.23.1,
peft 0.19.1, datasets 4.0.0, accelerate 1.10.1, on one RTX 4090. Generation is greedy, but
batched bf16 kernels are not bit-exact across versions or runs — expect individual ASR
points to move by about ±0.02 (the same attack measured 0.82 and 0.84 on two runs here).

**Pinned prompt date.** Llama-3.2's and Granite Guardian's chat templates stamp *today's*
date into the system prompt, which made every prompt, activation, generation and judge
score depend on the run date. All rendering now goes through `src/refusal_dpo/chat.py`,
which pins it to 08 Jul 2026 — the date of the committed preference pairs and every
Milestone 1 number, whose prompts it reproduces byte-exact. The Milestone 2 and seed
numbers above come from a pinned-date run. Earlier, unpinned runs (5–6 Aug and 27 Sep 2026)
found the same layers (L21, L15) and restored rank-1 ASR within ±0.02, but the base model's
curve moved by up to 0.12 with the date string alone — the reason its "min rank" (2–4) is
not reported as a result.

**Committed preference pairs are reused.** The *rejected* side of the harmful pairs is sampled
(T = 0.7), and the published pairs predate `build_prefs`' `--seed`, so rebuilding them
yields a different training set — and different DPO models — every time (an unseeded
rebuild shared 0 of 341 pairs with the committed set). `run_all.sh` and `run_balanced.sh`
therefore reuse the committed `data/prefs.jsonl`, `data/prefs_benign.jsonl` and
`data/prefs_balanced.jsonl` when present, so a re-run retrains on exactly the published
data. `REBUILD_PREFS=1` regenerates them instead; sampling is now seeded (`--seed`, default 0),
so rebuilt pairs are deterministic run-to-run, though not the published set.

```bash
# 0. install + fetch data. data/ is already committed, so this step is optional — skip it
#    to reproduce on the exact prompt pools used here.
pip install -r requirements.txt && pip install -e . --no-deps   # exact tested versions
huggingface-cli login                       # Llama-3.2 is gated — request access first
python scripts/prepare_data.py --out-dir data \
     --val-source harmbench-standard --eval-source harmbench-standard

# 1. base pipeline -> jailbroken ckpt, harmful prefs, harmful-only DPO, eval + re-attack
#    (produces runs/jailbroken, data/prefs.jsonl, runs/eval_*.json, runs/reattack_*.json)
PYTHON=/path/to/venv/bin/python bash scripts/run_all.sh \
  data/harmful_extract.txt data/harmful_val.txt data/harmful_train.txt \
  data/harmless.txt data/benign.txt data/harmful_eval.txt

# 2. the winning recipe: balanced (tax-free) restore -> eval -> re-attack -> figures ->
#    robustness check. Produces runs/dpo_balanced_2ep/merged and runs/fig_*.png.
PYTHON=/path/to/venv/bin/python bash scripts/run_balanced.sh
```

`run_balanced.sh` chains the six steps that turn the harmful-only baseline into the
tax-free restored model: harmful-only 2-epoch DPO (the over-refusing model that supplies
on-policy benign rejects) → benign comply-pairs → balanced DPO, 1 and 2 epochs + merge →
evaluate → re-attack in the base model's order → re-attack at the restored model's own
layers + figures. Each step's output feeds the next, so a fresh clone with no `runs/`
reproduces every row of both tables and both figures. Total ≈ 2–2.5 h on a 4090; the seed
repeats (`run_seed_check.sh`) add ≈ 40 min per seed.

Artifacts regenerated: `runs/eval_*.json` (frontier), `runs/reattack_*.json` (rank curves),
`runs/fig_*.png` (figures), `runs/dpo_balanced_2ep/merged` (restored model),
`artifacts/refusal_dir*.pt` (extracted directions).

### Reproducing the corrected Milestone 2

`scripts/run_reattack.sh` and `scripts/run_all.sh` derive the base model's candidate layers
from its fresh extraction (top 6 by refusal-under-ablation on val — `21 20 22 15 9 11` on
the committed data) and write to `runs/reattack_*_fixed.json`, so the superseded pre-fix
JSONs stay untouched; `scripts/plot_results.py` reads the corrected
files. The own-layer check (step 6 of `run_balanced.sh`, `scripts/run_ownlayer_check.sh`)
re-selects the restored model's layers from a fresh extraction and writes
`artifacts/refusal_dir_restored_fixed.pt`, `runs/reattack_restored_ownlayer_fixed.json` (rank-1
at the best layer) and `runs/reattack_restored_ownorder_fixed.json` (rank-1…6 in its own top-6
order, derived from the extraction scores). The seed repeats are
`SEEDS="1 2" bash scripts/run_seed_check.sh` (~40 min/seed on a 4090; needs `--seed`, now a
`train_dpo.py` flag). To run the Milestone 2 steps directly:

```bash
PY=/path/to/venv/bin/python
export PYTHONPATH=src HF_HOME=...

# corrected layer sweep on the base model -> selects L21
$PY -m refusal_dpo.direction.extract --model meta-llama/Llama-3.2-3B-Instruct \
  --harmful data/harmful_extract.txt --harmless data/harmless.txt --n-extract 128 \
  --harmful-val data/harmful_val.txt --n-val 100 --out artifacts/refusal_dir_fixed.pt

# each model's own top-6 layer order, derived from its extraction exactly as the scripts do
# (restored: artifacts/refusal_dir_restored_fixed.pt comes from run_ownlayer_check.sh step A)
order() { $PY -c "import torch; s=torch.load('$1', weights_only=False)['scores']; \
print(*sorted(s, key=lambda l: (s[l], l))[:6])"; }
BASE_LAYERS=$(order artifacts/refusal_dir_fixed.pt)             # -> 21 20 22 15 9 11
OWN_LAYERS=$(order artifacts/refusal_dir_restored_fixed.pt)     # -> 15 16 22 20 21 13

# corrected rank curves in the BASE model's order (~10 min for both models on a 4090)
for M in "meta-llama/Llama-3.2-3B-Instruct:original_fixed" \
         "runs/dpo_balanced_2ep/merged:restored_balanced_2ep_fixed"; do
  $PY -m refusal_dpo.eval.reattack --model "${M%%:*}" \
    --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
    --harmless data/harmless.txt --layers $BASE_LAYERS --max-rank 6 --n-eval 100 \
    --judge granite --tag "${M##*:}" --out "runs/reattack_${M##*:}.json"
done

# restored model attacked in its OWN layer order -> the headline restored curve
$PY -m refusal_dpo.eval.reattack --model runs/dpo_balanced_2ep/merged \
  --harmful-extract data/harmful_extract.txt --harmful-eval data/harmful_eval.txt \
  --harmless data/harmless.txt --layers $OWN_LAYERS --max-rank 6 --n-eval 100 \
  --judge granite --tag restored_ownorder_fixed --out runs/reattack_restored_ownorder_fixed.json
```

Note the layer sweep is stochastic only in generation (greedy, so deterministic), but the
extracted direction now depends on the *batch composition* only through the fixed
`batch_size=16` — with the indexing fix the read position no longer varies with batch
contents, so extraction is reproducible across batch sizes.