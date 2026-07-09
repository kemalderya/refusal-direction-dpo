"""
Standard eval: the safety/helpfulness frontier + capability tax.

  - ASR on harmful (refusal rate's complement): should be low pre-jailbreak,
    high after ablation, low again after DPO restoration.
  - over-refusal on benign (XSTest-safe / OR-Bench-benign): the alignment tax —
    DPO must restore refusal WITHOUT over-refusing. Reuses your project #2 sets.
  - MMLU retention (optional): the capability tax.

Run this on original / jailbroken / DPO-restored to fill the base->jailbroken->
restored comparison. Pass a HELD-OUT (cross-dataset) harmful set as --harmful so ASR
is not measured on prompts used for extraction / DPO.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_dpo.direction.extract import chat_prompt
from refusal_dpo.data.refusal_judge import is_refusal, complied
from refusal_dpo.device import best_device, model_dtype


@torch.no_grad()
def _gen(model, tok, prompts, bs=16, max_new=64):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = [chat_prompt(tok, p) for p in prompts[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(gen.size(0)):
            outs.append(tok.decode(gen[j, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--harmful", required=True, help="held-out (cross-dataset) harmful behaviors")
    ap.add_argument("--benign", required=True, help="XSTest/OR-Bench safe prompts")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--judge", choices=["substring", "granite"], default="substring",
                    help="ASR judge: substring proxy (fast) or granite harm classifier (headline)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = best_device()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=model_dtype(dev)).to(dev).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()

    harmful = _load(a.harmful); benign = _load(a.benign)
    h_out = _gen(model, tok, harmful)
    b_out = _gen(model, tok, benign)
    del model                                    # free the target before judging
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # over-refusal is a REFUSAL signal on benign prompts -> substring always (Granite
    # judges harm, not refusal). ASR is the harm/compliance signal on harmful prompts.
    over_refusal = sum(is_refusal(t) for t in b_out) / len(b_out)  # alignment tax
    if a.judge == "granite":
        from refusal_dpo.data.granite_judge import harmful_batch
        asr = sum(harmful_batch(list(zip(harmful, h_out)))) / len(h_out)  # harm-classified
    else:
        asr = sum(complied(t) for t in h_out) / len(h_out)               # substring proxy
    m = {"tag": a.tag, "judge": a.judge, "asr_harmful": asr,
         "over_refusal_benign": over_refusal, "refusal_harmful": 1 - asr}
    print(json.dumps(m, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(m, indent=2))


def _load(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return [r.get("instruction") or r.get("prompt") or r.get("text") for r in rows]
    return [l for l in p.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    main()
