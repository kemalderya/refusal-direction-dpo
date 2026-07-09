"""
Device / dtype resolution.

The project targets a single CUDA 4090 in bf16. Off-GPU (CPU, or Apple MPS) we
degrade to fp32 so the pipeline still *runs* for smoke tests / CI — bf16 on CPU is
unusably slow and only partially supported. It won't be fast off-GPU; it just won't
crash on `.to("cuda")`.
"""
from __future__ import annotations
import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str | None = None) -> torch.dtype:
    """bf16 on CUDA (the target); fp32 everywhere else."""
    return torch.bfloat16 if (device or best_device()) == "cuda" else torch.float32
