"""
Render the two headline figures from the JSON/eval artifacts:
  runs/fig_reattack.png  — Milestone 2: rank-vs-ASR re-attack (the finding)
  runs/fig_frontier.png  — Milestone 1: ASR vs over-refusal safety frontier (alignment tax)

Colors: Okabe-Ito CVD-safe categorical hues. One y-axis per chart, recessive grid,
direct end-labels, text in ink tokens (not the series color).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# Okabe-Ito CVD-safe hues
VERMILLION = "#D55E00"   # original / base
BLUE       = "#0072B2"   # restored (balanced-2ep)
GREEN      = "#009E73"   # balanced restored group
GRAY       = "#6b7280"   # baselines / reference
INK        = "#1f2937"   # text token
MUTED      = "#9aa1ab"   # grid / recessive

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.9,
})


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def reattack_fig():
    # *_fixed.json = the corrected curves (post last-token indexing fix in hooks.py).
    # The un-suffixed files hold the SUPERSEDED pre-fix curves; see RESULTS.md.
    #
    # Each model is attacked in its OWN layer order (top-6 layers by single-direction
    # refusal-under-ablation on held-out val). The restored model is also shown under the
    # base model's order (dashed) — that curve looked "higher-rank", but only because the
    # base order puts restored's decisive layer (15) fourth.
    orig_j = json.loads((RUNS / "reattack_original_fixed.json").read_text())
    orig, orig_layers = orig_j["rank_asr_curve"], orig_j["layers"]
    rest_base = json.loads((RUNS / "reattack_restored_balanced_2ep_fixed.json").read_text())["rank_asr_curve"]
    own_path = RUNS / "reattack_restored_ownorder_fixed.json"
    if own_path.exists():
        own_j = json.loads(own_path.read_text())
        rest_own, own_layers = own_j["rank_asr_curve"], own_j["layers"]
    else:  # full own-order curve not run yet -> just the rank-1 own-layer (L15) point
        own_j = json.loads((RUNS / "reattack_restored_ownlayer_fixed.json").read_text())
        rest_own, own_layers = own_j["rank_asr_curve"], own_j["layers"]
    # no-attack (rank-0) ASR from the eval artifacts
    a0_orig = json.loads((RUNS / "eval_original.json").read_text())["asr_harmful"]
    a0_rest = json.loads((RUNS / "eval_restored_balanced_2ep.json").read_text())["asr_harmful"]

    ranks = [int(k) for k in sorted(orig, key=int)]
    own_ranks = [int(k) for k in sorted(rest_own, key=int)]
    xo = [0] + ranks
    yo = [a0_orig] + [orig[str(k)] for k in ranks]
    yb = [a0_rest] + [rest_base[str(k)] for k in ranks]
    xr = [0] + own_ranks
    yr = [a0_rest] + [rest_own[str(k)] for k in own_ranks]

    # first rank at/above the 0.80 jailbreak threshold, per model (own-order attacks)
    kx_o = next((k for k in ranks if orig[str(k)] >= 0.80), None)
    kx_r = next((k for k in own_ranks if rest_own[str(k)] >= 0.80), None)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    # jailbreak threshold
    ax.axhline(0.80, ls=(0, (4, 4)), lw=1.1, color=GRAY, zorder=1,
               label="jailbreak threshold (0.80)")

    ax.plot(xo, yo, "-o", color=VERMILLION, lw=2, ms=7, zorder=3,
            label=f"original — own layer order (L{', '.join(map(str, orig_layers[:3]))}, …)")
    ax.plot(xo, yb, "--o", color=BLUE, lw=1.6, ms=6, alpha=0.45, zorder=2,
            label="restored — base model's layer order")
    ax.plot(xr, yr, "-o", color=BLUE, lw=2, ms=7, zorder=3,
            label=f"restored — own layer order (L{', '.join(map(str, own_layers[:3]))}, …)")

    # ring the rank at which each model first crosses the threshold
    for k, curve, c in ((kx_o, orig, VERMILLION), (kx_r, rest_own, BLUE)):
        if k is not None:
            ax.scatter([k], [curve[str(k)]], s=230, facecolors="none",
                       edgecolors=c, linewidths=1.8, zorder=4)

    # direct end-labels
    # (the three curves can end within 0.01 of each other: stack labels top-down by end value,
    #  at least GAP apart, centred on the curves' mean end value)
    ends = [(yb[-1], " restored (base order)", "normal"), (yo[-1], " original", "bold")]
    if xr[-1] == 6:
        ends.append((yr[-1], " restored (own order)", "bold"))
    ends.sort(key=lambda e: -e[0])
    GAP = 0.05
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = min(ys[i], ys[i - 1] - GAP)
    shift = (sum(e[0] for e in ends) - sum(ys)) / len(ys)
    for (y0, text, weight), y in zip(ends, ys):
        ax.text(6.02, y + shift, text, va="center", color=INK, fontsize=9.5, fontweight=weight)

    if kx_r is not None:
        ax.annotate(f"restored jailbreaks at rank {kx_r}\n(its own layer 15)",
                    xy=(kx_r, rest_own[str(kx_r)]), xytext=(1.35, 0.95), va="center", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
    if kx_o is not None:
        ax.annotate(f"original needs rank {kx_o}", xy=(kx_o, orig[str(kx_o)]),
                    xytext=(3.25, 0.5), fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    ax.set_xlabel("rank of ablated subspace  (# directions removed;  0 = no attack)")
    ax.set_ylabel("attack success rate  (Granite, n=100 held-out)")
    ax.set_title("Re-attack: DPO relocates refusal — one direction still breaks it"
                 if kx_r == 1 else "Re-attack: rank-vs-ASR, original vs DPO-restored")
    ax.set_xlim(-0.2, 7.05)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(0, 7))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color=MUTED, alpha=0.28, lw=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    _despine(ax)
    fig.tight_layout()
    out = RUNS / "fig_reattack.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def frontier_fig():
    # (label, eval JSON tag, group)  group: base | tax | good — read from runs/eval_<tag>.json
    states = [
        ("original",         "original",              "base"),
        ("jailbroken",       "jailbroken",            "base"),
        ("harmful-only 1ep", "restored_1ep",          "tax"),
        ("harmful-only 2ep", "restored_2ep",          "tax"),
        ("balanced 1ep",     "restored_balanced_1ep", "good"),
        ("balanced 2ep",     "restored_balanced_2ep", "good"),
    ]
    pts = []
    for name, tag, grp in states:
        ev = json.loads((RUNS / f"eval_{tag}.json").read_text())
        pts.append((name, ev["asr_harmful"], ev["over_refusal_benign"], grp))
    color = {"base": GRAY, "tax": VERMILLION, "good": BLUE}
    legend_name = {"base": "baseline states", "tax": "harmful-only DPO (alignment tax)",
                   "good": "balanced DPO (tax fixed)"}

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    seen = set()
    for name, asr, orf, grp in pts:
        lbl = legend_name[grp] if grp not in seen else None
        seen.add(grp)
        ax.scatter(asr, orf, s=90, color=color[grp], zorder=3, label=lbl,
                   edgecolor="white", linewidth=1.2)
    # point labels — (dx, dy, ha) in offset points; ideal corner is bottom-left
    offsets = {
        "original":         (-7,  9, "right"),
        "jailbroken":       (0,  11, "center"),
        "harmful-only 1ep": (10,  0, "left"),
        "harmful-only 2ep": (10,  0, "left"),
        "balanced 1ep":     (8,   8, "left"),
        "balanced 2ep":     (8, -14, "left"),
    }
    for name, asr, orf, grp in pts:
        dx, dy, ha = offsets[name]
        ax.annotate(name, (asr, orf), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9, color=INK, ha=ha)

    # the ideal corner (low ASR, low over-refusal)
    ax.annotate("← safer     more helpful ↓\nideal corner", xy=(0.045, 0.14),
                fontsize=9.5, color=GREEN, fontweight="bold", ha="left", va="center")
    ax.set_xlabel("attack success rate on harmful  (lower = safer)")
    ax.set_ylabel("over-refusal on benign  (lower = more helpful)")
    ax.set_title("Safety frontier: contrastive data removes the DPO alignment tax")
    ax.set_xlim(-0.03, 0.9)
    ax.set_ylim(-0.06, 1.1)
    ax.grid(color=MUTED, alpha=0.28, lw=0.8)
    ax.legend(loc="center right", frameon=False, fontsize=9.5)
    _despine(ax)
    fig.tight_layout()
    out = RUNS / "fig_frontier.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    reattack_fig()
    frontier_fig()
