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
    orig = json.loads((RUNS / "reattack_original.json").read_text())["rank_asr_curve"]
    rest = json.loads((RUNS / "reattack_restored_balanced_2ep.json").read_text())["rank_asr_curve"]
    # no-attack (rank-0) ASR from the eval artifacts
    a0_orig = json.loads((RUNS / "eval_original.json").read_text())["asr_harmful"]
    a0_rest = json.loads((RUNS / "eval_restored_balanced_2ep.json").read_text())["asr_harmful"]

    ranks = [int(k) for k in sorted(orig, key=int)]
    xo = [0] + ranks
    yo = [a0_orig] + [orig[str(k)] for k in ranks]
    yr = [a0_rest] + [rest[str(k)] for k in ranks]

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    # jailbroken reference ceiling
    ax.axhline(0.80, ls=(0, (4, 4)), lw=1.1, color=GRAY, zorder=1)
    ax.text(3.0, 0.845, "jailbroken ceiling (0.80)", va="bottom", ha="center",
            fontsize=9, color=GRAY)

    ax.plot(xo, yo, "-o", color=VERMILLION, lw=2, ms=7, zorder=3, label="original")
    ax.plot(xo, yr, "-o", color=BLUE, lw=2, ms=7, zorder=3, label="restored (balanced-2ep)")

    # direct end-labels
    ax.text(6.02, yo[-1] - 0.015, " original", va="center", color=VERMILLION, fontsize=10, fontweight="bold")
    ax.text(6.02, yr[-1] + 0.005, " restored", va="center", color=BLUE, fontsize=10, fontweight="bold")

    # annotate the two decisive points
    ax.annotate("rank-1 suffices\n(0.83)", xy=(1, yo[1]), xytext=(1.25, 0.90),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
    ax.annotate("rank-1 blunted (0.60);\nclimbs with rank, never ≥0.8",
                xy=(1, yr[1]), xytext=(1.5, 0.40), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    ax.set_xlabel("rank of ablated subspace  (# directions removed;  0 = no attack)")
    ax.set_ylabel("attack success rate  (Granite, 100 held-out harmful)")
    ax.set_title("Re-attack: refusal is rank-1 in the base model, distributed after DPO")
    ax.set_xlim(-0.2, 6.7)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(0, 7))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color=MUTED, alpha=0.28, lw=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    _despine(ax)
    fig.tight_layout()
    out = RUNS / "fig_reattack.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def frontier_fig():
    # (label, ASR, over_refusal, group)  group: base | tax | good
    pts = [
        ("original",            0.18, 0.032, "base"),
        ("jailbroken",          0.80, 0.012, "base"),
        ("harmful-only 1ep",    0.19, 0.828, "tax"),
        ("harmful-only 2ep",    0.19, 1.000, "tax"),
        ("balanced 1ep",        0.37, 0.004, "good"),
        ("balanced 2ep",        0.24, 0.004, "good"),
    ]
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
