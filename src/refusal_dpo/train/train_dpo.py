"""
Phase 3 — restore refusal with DPO.

This is the CV-critical step: real preference optimization, not imitation.
Key mechanics:

  loss = -log σ( β·[logπ_θ(chosen) - logπ_ref(chosen)]
                 - β·[logπ_θ(rejected) - logπ_ref(rejected)] )

  - β (~0.1) is an inverse-temperature / KL leash to the reference.
  - With LoRA, π_ref = base model (adapters DISABLED), π_θ = base + adapters. TRL's
    DPOTrainer does this adapter-toggle automatically when ref_model=None + PEFT, so
    we hold ONE set of base weights, not two. That's what fits a 3B DPO on a 4090.
  - WATCH the known DPO pathology: the loss only cares about the *gap*, so both
    chosen and rejected logprobs can fall together. TRL logs rewards/chosen and
    rewards/rejected — watch that chosen doesn't crater in absolute terms.

We DPO the JAILBROKEN (orthogonalized) checkpoint back toward refusal.

API note: TRL churns. Written against TRL >=0.12 (DPOConfig + DPOTrainer,
processing_class=). If your TRL differs, the args names are the only thing to fix.
"""
from __future__ import annotations
import argparse, yaml
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from refusal_dpo.device import best_device, model_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="prefs.jsonl from build_prefs")
    ap.add_argument("--model", default=None, help="checkpoint to DPO (the jailbroken one)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (e.g. sweep 1 vs 2)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed LoRA init + data order (omit = original runs: DPOConfig default 42)")
    a = ap.parse_args()
    if a.seed is not None:
        set_seed(a.seed)  # before the model/LoRA are built — DPOTrainer wraps PEFT pre-seed
    cfg = yaml.safe_load(Path(a.config).read_text())
    model_id = a.model or cfg["model_id"]
    out_dir = a.out or cfg["output_dir"]

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = best_device()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_dtype(dev))

    ds = load_dataset("json", data_files=a.data, split="train")  # prompt/chosen/rejected

    peft_cfg = LoraConfig(
        r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05), bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules",
                               ["q_proj", "k_proj", "v_proj", "o_proj"]))

    args = DPOConfig(
        output_dir=out_dir,
        beta=cfg.get("beta", 0.1),                 # the KL leash / inverse temp
        learning_rate=float(cfg.get("lr", 5e-5)),
        num_train_epochs=(a.epochs if a.epochs is not None else cfg.get("epochs", 1)),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        max_length=cfg.get("max_length", 1024),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=10, save_strategy="epoch",
        bf16=(dev == "cuda"), report_to=[],
        **({"seed": a.seed} if a.seed is not None else {}),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # None + PEFT -> reference is the adapter-disabled base
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(f"{out_dir}/adapter")
    tok.save_pretrained(f"{out_dir}/adapter")
    print(f"DPO adapter -> {out_dir}/adapter")
    print("Merge for the re-attack (plain weights): scripts/merge_adapter.py")


if __name__ == "__main__":
    main()
