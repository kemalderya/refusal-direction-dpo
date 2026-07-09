"""Merge a DPO LoRA adapter into base weights (bf16) for a clean re-attack checkpoint."""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base-model", required=True, help="the jailbroken checkpoint DPO trained on")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
merged.save_pretrained(a.out)
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
print(f"merged DPO-restored checkpoint -> {a.out}")
