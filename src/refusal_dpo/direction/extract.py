"""
Phase 1 — extract the refusal direction (diff-in-means) and select the layer.

r_ℓ = mean(harmful acts)_ℓ − mean(harmless acts)_ℓ , normalized.

Layer selection: the best direction is the one whose ablation most kills refusal.
We score candidate layers by "refusal drop under ablation" on a HELD-OUT harmful set,
disjoint from the prompts the direction was fit on, so the selected layer isn't
validated on its own fitting data. Prefer a cross-dataset val set via `--harmful-val`
(e.g. HarmBench while fitting on AdvBench) — a different distribution makes the layer
choice generalize; without it we fall back to carving a `--val-frac` slice off
`--harmful`. (A quick proxy — refusal-string rate — is enough here; upgrade to an LLM
judge for the final number.)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from refusal_dpo.model.hooks import collect_last_token_acts, ablate_direction
from refusal_dpo.data.refusal_judge import is_refusal
from refusal_dpo.device import best_device, model_dtype


def chat_prompt(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True)


def diff_in_means(harmful_acts, harmless_acts):
    """[N,L+1,d],[M,L+1,d] -> [L+1,d] normalized per layer."""
    d = harmful_acts.mean(0) - harmless_acts.mean(0)     # [L+1, d]
    return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def _refusal_rate_under_ablation(model, tok, prompts, direction, n=None):
    hits = 0
    used = prompts if n is None else prompts[:n]
    for p in used:
        enc = tok(chat_prompt(tok, p), return_tensors="pt",
                  add_special_tokens=False).to(model.device)
        with ablate_direction(model, direction):
            out = model.generate(**enc, max_new_tokens=40, do_sample=False)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        hits += int(is_refusal(text))
    return hits / max(1, len(used))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--harmful", required=True, help="AdvBench harmful instructions (.txt/.jsonl)")
    ap.add_argument("--harmless", required=True, help="Alpaca harmless instructions (.txt/.jsonl)")
    ap.add_argument("--out", default="artifacts/refusal_dir.pt")
    ap.add_argument("--n-extract", type=int, default=256)
    ap.add_argument("--harmful-val", default=None,
                    help="separate held-out harmful set for LAYER SELECTION (e.g. HarmBench). "
                         "If given, the FULL --harmful set fits the direction (no internal split).")
    ap.add_argument("--n-val", type=int, default=100,
                    help="how many --harmful-val prompts to score during layer selection")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fallback: fraction of --harmful held out for layer selection "
                         "when --harmful-val is not given")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = best_device()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=model_dtype(dev)).to(dev).eval()

    harmful = _load(a.harmful)[: a.n_extract]
    harmless = _load(a.harmless)[: a.n_extract]

    # Fit the direction on `harmful_fit`, select the layer on the disjoint `harmful_val`.
    if a.harmful_val:
        # Cross-dataset held-out val (e.g. HarmBench): fit on ALL of --harmful, no carve-out.
        harmful_fit = harmful
        harmful_val = _load(a.harmful_val)[: a.n_val]
        print(f"fit on {len(harmful_fit)} harmful / {len(harmless)} harmless; layer selection "
              f"on {len(harmful_val)} held-out prompts from {a.harmful_val}")
    else:
        # Fallback: carve a val slice off --harmful. Degrade gracefully if the input is tiny.
        n_val = max(8, int(len(harmful) * a.val_frac))
        if len(harmful) > n_val:
            harmful_fit, harmful_val = harmful[:-n_val], harmful[-n_val:]
        else:
            harmful_fit, harmful_val = harmful, harmful
            print(f"WARN: only {len(harmful)} harmful prompts; layer selection is NOT held out.")

    hp = [chat_prompt(tok, x) for x in harmful_fit]
    lp = [chat_prompt(tok, x) for x in harmless]
    ha = collect_last_token_acts(model, tok, hp)
    la = collect_last_token_acts(model, tok, lp)
    dirs = diff_in_means(ha, la)                 # [L+1, d]

    # Select the layer whose direction most reduces refusal when ablated, scored on
    # the held-out slice. (Skip embeddings=0 and the last couple of layers; mid-stack
    # usually wins.)
    L = dirs.size(0)
    candidates = range(int(L * 0.3), int(L * 0.8))
    scores = {}
    for l in candidates:
        rr = _refusal_rate_under_ablation(model, tok, harmful_val, dirs[l])
        scores[l] = rr
        print(f"layer {l:2d}: refusal-rate-under-ablation (held-out) = {rr:.3f}")
    best = min(scores, key=scores.get)           # lowest refusal after ablation
    print(f"selected layer {best} (refusal {scores[best]:.3f} under ablation)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": dirs[best], "layer": best,
                "all_layer_dirs": dirs, "scores": scores,
                "n_fit": len(harmful_fit), "n_harmless": len(harmless),
                "n_val": len(harmful_val),
                "val_source": a.harmful_val or f"internal-split(val_frac={a.val_frac})"}, a.out)
    print(f"saved -> {a.out}")


def _load(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return [r.get("instruction") or r.get("prompt") or r.get("text") for r in rows]
    return [l for l in p.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    main()
