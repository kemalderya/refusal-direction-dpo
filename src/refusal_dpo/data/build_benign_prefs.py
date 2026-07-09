"""
Phase 3 data (BENIGN half) — comply-pairs that stop DPO from over-refusing.

The harmful pairs (`build_prefs.py`) only ever say "prefer refusal over compliance",
so DPO trained on them alone raises refusal GLOBALLY and over-refuses benign prompts
(the alignment tax). These benign pairs supply the missing contrast:

  chosen   = a helpful answer            (from the ORIGINAL model — it complies on benign)
  rejected = an over-refusal              (from the RESTORED/over-refusing model — ON-POLICY
             negative, mirroring how the harmful pairs used the jailbroken model's comply)

Keep a pair only if chosen actually complied AND rejected actually refused, so every pair
is a clean comply-vs-refuse contrast. Concatenate with the harmful pairs and DPO learns to
refuse harmful AND comply with benign.

Benign prompts should be DISJOINT from the XSTest over-refusal eval (use OR-Bench-hard +
extra Alpaca via prepare_data's benign_train.txt).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_dpo.data.build_prefs import generate, chat_prompt, _load
from refusal_dpo.data.refusal_judge import is_refusal, complied
from refusal_dpo.device import best_device, model_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--refusal-model", required=True,
                    help="over-refusing checkpoint (source of on-policy rejected refusals)")
    ap.add_argument("--benign", required=True, help="benign training prompts (benign_train.txt)")
    ap.add_argument("--out", default="data/prefs_benign.jsonl")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.orig_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    benign = _load(a.benign)[: a.n]
    dev = best_device()
    dtype = model_dtype(dev)

    orig = AutoModelForCausalLM.from_pretrained(
        a.orig_model, torch_dtype=dtype).to(dev).eval()
    chosen = generate(orig, tok, benign, sample=False)     # helpful answers (greedy)
    del orig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ref = AutoModelForCausalLM.from_pretrained(
        a.refusal_model, torch_dtype=dtype).to(dev).eval()
    rejected = generate(ref, tok, benign, sample=False)    # on-policy over-refusals (greedy)
    del ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows, kept = [], 0
    for instr, ch, rj in zip(benign, chosen, rejected):
        if complied(ch) and is_refusal(rj):                # clean comply-vs-refuse only
            rows.append({"prompt": chat_prompt(tok, instr), "chosen": ch, "rejected": rj})
            kept += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"kept {kept}/{len(benign)} clean benign comply-pairs -> {a.out}")
    if kept < 0.3 * len(benign):
        print("NOTE: low benign yield. Either the refusal-model doesn't over-refuse these "
              "prompts (pick a more over-refusing ckpt) or the original refuses them too.")


if __name__ == "__main__":
    main()
