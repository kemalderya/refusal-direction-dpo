"""
Phase 2 — bake the ablation into the weights (Arditi weight orthogonalization).

Runtime ablation (hooks) is great for measurement but awkward to DPO from. Instead
we orthogonalize every matrix that WRITES to the residual stream against r̂, so the
model can never produce an r̂ component — a permanently jailbroken checkpoint we can
fine-tune normally.

For a matrix W whose output is the residual (o_proj, down_proj: [d_model, in]):
    W ← W − r̂ (r̂ᵀ W)          # kill the r̂ row-space component
For embed_tokens ([vocab, d_model], each row is a residual vector):
    W ← W − (W r̂) r̂ᵀ          # kill r̂ from each embedding row

Orthogonalization is pure weight math and runs fine on CPU; we keep bf16 so the
saved checkpoint is ready to load on the 4090.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def orthogonalize_(model, direction: torch.Tensor):
    r = (direction / direction.norm()).float()

    def kill_output(W):  # W: [d_model, in], output dim = rows
        r_ = r.to(W.device)
        return (W.float() - torch.outer(r_, r_ @ W.float())).to(W.dtype)

    def kill_rows(W):    # W: [vocab, d_model], each row is a residual vector
        r_ = r.to(W.device)
        return (W.float() - torch.outer(W.float() @ r_, r_)).to(W.dtype)

    model.model.embed_tokens.weight.data = kill_rows(model.model.embed_tokens.weight.data)
    for layer in model.model.layers:
        layer.self_attn.o_proj.weight.data = kill_output(layer.self_attn.o_proj.weight.data)
        layer.mlp.down_proj.weight.data = kill_output(layer.mlp.down_proj.weight.data)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--direction", required=True, help="refusal_dir.pt from extract.py")
    ap.add_argument("--out", required=True, help="output dir for the jailbroken checkpoint")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    d = torch.load(a.direction)["direction"]
    orthogonalize_(model, d)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"jailbroken (orthogonalized) checkpoint -> {a.out}")
    print("Sanity-check it complies on a few harmful prompts before DPO-ing it back.")


if __name__ == "__main__":
    main()
