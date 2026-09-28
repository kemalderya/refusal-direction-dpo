"""
Phase 3 data — DPO preference pairs to restore refusal.

For each harmful instruction:
  chosen   = a refusal  (sampled from the ORIGINAL aligned model — natural refusals)
  rejected = a harmful compliance (sampled from the JAILBROKEN model — ON-POLICY
             negatives, which is exactly when DPO works best; see the DPO on-policy
             point in PROJECT notes)

We keep a pair only if chosen actually refused AND rejected actually complied, so
every pair is a clean refuse-vs-comply contrast. Output: JSONL of
{prompt, chosen, rejected} ready for TRL DPOTrainer.

Use the TRAIN split of harmful here; the held-out (cross-dataset) set is reserved
for eval/reattack so the restored model is never measured on prompts it trained on.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from refusal_dpo.data.refusal_judge import is_refusal, complied
from refusal_dpo.device import best_device, model_dtype
from refusal_dpo.chat import chat_prompt


@torch.no_grad()
def generate(model, tok, prompts, max_new=128, sample=True, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = [chat_prompt(tok, p) for p in prompts[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=sample, temperature=0.7 if sample else None,
                             top_p=0.9 if sample else None,
                             pad_token_id=tok.pad_token_id)
        for j in range(gen.size(0)):
            outs.append(tok.decode(gen[j, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--jailbroken", required=True, help="orthogonalized checkpoint dir")
    ap.add_argument("--harmful", required=True, help="TRAIN harmful instructions")
    ap.add_argument("--out", default="data/prefs.jsonl")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the sampled (rejected) generations so pairs rebuild identically")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.orig_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    harmful = _load(a.harmful)[: a.n]
    dev = best_device()
    dtype = model_dtype(dev)

    orig = AutoModelForCausalLM.from_pretrained(
        a.orig_model, torch_dtype=dtype).to(dev).eval()
    chosen = generate(orig, tok, harmful, sample=False)   # natural refusals
    del orig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    jb = AutoModelForCausalLM.from_pretrained(
        a.jailbroken, torch_dtype=dtype).to(dev).eval()
    set_seed(a.seed)  # rejected is SAMPLED (T=0.7) — unseeded, every rebuild gave different pairs
    rejected = generate(jb, tok, harmful, sample=True)    # on-policy compliances

    rows, kept = [], 0
    for instr, ch, rj in zip(harmful, chosen, rejected):
        if is_refusal(ch) and complied(rj):               # clean contrast only
            rows.append({"prompt": chat_prompt(tok, instr), "chosen": ch, "rejected": rj})
            kept += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"kept {kept}/{len(harmful)} clean pairs -> {a.out}")
    if kept < 0.5 * len(harmful):
        print("NOTE: low yield. Either the jailbreak is weak (check orthogonalization) "
              "or the original refuses inconsistently (raise n, or template chosen).")


def _load(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return [r.get("instruction") or r.get("prompt") or r.get("text") for r in rows]
    return [l for l in p.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    main()
