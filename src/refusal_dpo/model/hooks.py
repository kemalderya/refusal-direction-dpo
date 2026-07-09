"""
Residual-stream tooling: capture activations (for direction extraction) and ablate
a direction (for the jailbreak / re-attack).

Two mechanisms, deliberately different:
  - CAPTURE uses output_hidden_states=True — no manual hooks, gives the residual
    stream at every layer in one forward pass. Used by direction/extract.py.
  - ABLATION uses forward hooks that project the residual off a direction at every
    layer + the embedding. Used at generation time (extract can't do that).

Directional ablation (Arditi et al.): for residual x and unit direction r̂,
    x ← x − (x · r̂) r̂
applied at EVERY layer and token position, so the r̂ component stays ~0 throughout
the stream. One direction, ablated everywhere — that's the whole intervention.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch


@torch.no_grad()
def collect_last_token_acts(model, tokenizer, prompts, batch_size=16, device=None):
    """
    Return residual-stream activations at the LAST token, all layers.
    Shape: [n_prompts, n_layers+1, d_model]  (index 0 = embeddings, then each layer).

    Prompts must already be chat-templated strings ending at the point where the
    model would begin its reply — that last position is where the refusal decision
    is encoded, which is why Arditi read it there.

    `device` defaults to the model's own device (portable across cuda/mps/cpu).
    """
    if device is None:
        device = next(model.parameters()).device
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple[L+1]
        # last non-pad position per row
        last = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(hs[0].size(0))
        # stack layers -> [B, L+1, d]
        acts = torch.stack([h[rows, last] for h in hs], dim=1)
        out.append(acts.float().cpu())
    return torch.cat(out, 0)


def _residual_writing_modules(model):
    """Layers whose OUTPUT is the residual stream — where ablation must be applied.
    For Llama: embed_tokens + every decoder layer. (Ablating decoder outputs keeps the
    r̂ component near zero as the stream accumulates.)"""
    mods = [model.model.embed_tokens]
    mods += list(model.model.layers)
    return mods


@contextmanager
def ablate_direction(model, direction: torch.Tensor):
    """
    Context manager: while active, ablate `direction` from the residual stream at
    every layer. Restores the model on exit. Use around .generate() to jailbreak.
    """
    r = (direction / direction.norm()).to(next(model.parameters()).dtype)
    r = r.to(next(model.parameters()).device)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: [B, T, d]; remove the r component at every position
        proj = (h @ r).unsqueeze(-1) * r
        h = h - proj
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    handles = [m.register_forward_hook(hook) for m in _residual_writing_modules(model)]
    try:
        yield model
    finally:
        for hdl in handles:
            hdl.remove()


@contextmanager
def ablate_subspace(model, directions: torch.Tensor):
    """
    Ablate a k-dim subspace (rows = directions). Used by reattack.py to measure how
    many directions must be removed to jailbreak (rank of the refusal subspace).
    `directions` is orthonormalized here before use.
    """
    Q, _ = torch.linalg.qr(directions.float().T)   # [d, k] orthonormal columns
    Q = Q.to(next(model.parameters()).dtype).to(next(model.parameters()).device)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h @ Q) @ Q.T          # project onto subspace
        h = h - proj                   # remove it
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    handles = [m.register_forward_hook(hook) for m in _residual_writing_modules(model)]
    try:
        yield model
    finally:
        for hdl in handles:
            hdl.remove()
