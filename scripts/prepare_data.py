"""
Fetch and write the five prompt files the pipeline expects (one prompt per line):

  harmful_extract.txt — AdvBench slice A  (direction fit + reattack re-extract)
  harmful_train.txt   — AdvBench slice B  (DPO preference pairs; DISJOINT from slice A)
  harmful_val.txt     — HarmBench (default)          (LAYER SELECTION; cross-dataset held-out)
  harmless.txt        — Alpaca instructions          (harmless contrast for diff-in-means)
  benign.txt          — XSTest safe prompts           (over-refusal / alignment tax)
  harmful_eval.txt    — a DIFFERENT harmful set       (HELD-OUT cross-dataset ASR + re-attack)

Two independence guarantees: (1) AdvBench is split into DISJOINT extract vs DPO pools, so
the refusal direction is never fit on the prompts DPO trains on (kills the "you re-extract
from the prompts you trained on" objection); (2) ASR + re-attack are scored on a distinct
cross-dataset harmful set, so the headline number reflects generalization, not memorization.

Sources are PUBLIC (canonical GitHub-raw CSVs + non-gated HF datasets) — the earlier
`walledai/*` HF mirrors are gated and 403 without per-dataset access. SEAM: if a URL/repo
moves or renames a column, edit SOURCES. Needs network; HF datasets need `datasets`.

    python scripts/prepare_data.py --out-dir data --eval-source malicious-instruct
"""
from __future__ import annotations
import argparse, io, csv, urllib.request
from pathlib import Path

_ADV = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
_XSTEST = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
_HARMBENCH = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
_MALICIOUS = "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt"

# kind: csv_url | txt_url | hf ; fields: candidate text columns (csv_url/hf only)
SOURCES = {
    "advbench": dict(kind="csv_url", url=_ADV, fields=["goal", "prompt", "behavior"]),
    "alpaca":   dict(kind="hf", repo="tatsu-lab/alpaca", split="train",
                     fields=["instruction"], drop_if_nonempty="input"),
    "xstest":   dict(kind="csv_url", url=_XSTEST, fields=["prompt"], keep_if_safe=True),
    # --- cross-dataset harmful eval options (pick one via --eval-source) ---------
    "jailbreakbench": dict(kind="hf", repo="JailbreakBench/JBB-Behaviors", config="behaviors",
                           split="harmful", fields=["Goal", "goal", "prompt"]),
    "harmbench": dict(kind="csv_url", url=_HARMBENCH, fields=["Behavior", "prompt"]),
    # HarmBench standard-only (200 safety-harm behaviors, no copyright/contextual). One pool
    # can serve BOTH val and eval: prepare_data splits it disjointly (val first, eval the rest).
    "harmbench-standard": dict(kind="csv_url", url=_HARMBENCH, fields=["Behavior"], keep_if_standard=True),
    "malicious-instruct": dict(kind="txt_url", url=_MALICIOUS),
    # OR-Bench-hard: "seemingly toxic but SAFE" prompts — benign DPO training data to
    # anchor compliance and prevent over-refusal (never used for eval; XSTest stays eval).
    "orbench-hard": dict(kind="hf", repo="bench-llm/or-bench", config="or-bench-hard-1k",
                         split="train", fields=["prompt"]),
}


def _get(url, timeout=60):
    return urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", "replace")


def _rows(spec):
    kind = spec["kind"]
    if kind == "txt_url":
        return [{"__text__": ln} for ln in _get(spec["url"]).splitlines()]
    if kind == "csv_url":
        return list(csv.DictReader(io.StringIO(_get(spec["url"]))))
    if kind == "hf":
        from datasets import load_dataset
        return (load_dataset(spec["repo"], spec["config"], split=spec["split"])
                if spec.get("config") else load_dataset(spec["repo"], split=spec["split"]))
    raise ValueError(f"unknown source kind: {kind}")


def _field(row, fields):
    for f in fields:
        v = row.get(f)
        if v:
            return str(v).strip()
    return None


def _looks_safe(row) -> bool:
    # XSTest labels safe vs unsafe prompts; keep only the safe (benign) ones.
    for key in ("label", "gpt4_label", "safety_label", "type"):
        v = str(row.get(key, "")).lower()
        if v in ("safe", "benign"):
            return True
        if v in ("unsafe", "harmful") or v.startswith("contrast"):
            return False
    return True  # no usable label -> keep (fallback)


def fetch(name, limit=None):
    spec = SOURCES[name]
    fields = spec.get("fields", ["__text__"])
    prompts, seen = [], set()
    for row in _rows(spec):
        row = dict(row)
        if spec.get("drop_if_nonempty") and str(row.get(spec["drop_if_nonempty"], "")).strip():
            continue
        if spec.get("keep_if_safe") and not _looks_safe(row):
            continue
        if spec.get("keep_if_standard") and str(row.get("FunctionalCategory", "")).lower() != "standard":
            continue  # HarmBench: drop copyright (IP, not safety-harm) + contextual (needs ContextString)
        p = _field(row, fields)
        if p:
            p = " ".join(p.split())  # collapse embedded newlines/whitespace -> one line per prompt
        if p and p not in seen:      # (Alpaca instructions sometimes span multiple lines)
            seen.add(p)
            prompts.append(p)
        if limit and len(prompts) >= limit:
            break
    return prompts


def _write(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines):5d} prompts -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--eval-source", default="harmbench-standard",
                    choices=["harmbench-standard", "malicious-instruct", "harmbench", "jailbreakbench"],
                    help="held-out harmful set for ASR / re-attack (auto-filtered disjoint from val/extract/train)")
    ap.add_argument("--val-source", default="harmbench-standard",
                    choices=["harmbench-standard", "harmbench", "jailbreakbench", "malicious-instruct"],
                    help="harmful set for LAYER SELECTION; MAY equal --eval-source (the pool is split "
                         "disjointly: val takes the first --n-val, eval takes the rest)")
    ap.add_argument("--n-val", type=int, default=100,
                    help="number of layer-selection prompts to take from the front of --val-source")
    ap.add_argument("--n-harmless", type=int, default=512)
    ap.add_argument("--n-benign-train", type=int, default=180,
                    help="benign DPO-training prompts to take from EACH of extra-Alpaca and "
                         "OR-Bench-hard (comply-pairs that counter over-refusal)")
    ap.add_argument("--n-harmful", type=int, default=None, help="cap total AdvBench (default: all)")
    ap.add_argument("--n-extract", type=int, default=128,
                    help="size of the DISJOINT extraction pool split off the front of AdvBench "
                         "(direction fit + reattack re-extract); the remainder is the DPO pool")
    a = ap.parse_args()
    out = Path(a.out_dir)

    # val_source MAY equal eval_source: the eval dedup below removes every val prompt, so the
    # shared pool is split into disjoint val (front) / eval (rest) halves — like AdvBench's split.

    # AdvBench, split into DISJOINT pools so extraction prompts are never reused for DPO
    # training or scoring: [:n_extract] -> extract/re-extract, [n_extract:] -> DPO pairs.
    adv = fetch("advbench", a.n_harmful)
    n_ex = min(a.n_extract, len(adv))
    harmful_extract, harmful_train = adv[:n_ex], adv[n_ex:]
    if not harmful_train:
        raise SystemExit(
            f"AdvBench yielded {len(adv)} prompts but --n-extract={a.n_extract} leaves none "
            f"for DPO. Lower --n-extract or raise --n-harmful.")

    # Layer-selection pool: keep it disjoint from the AdvBench fit/train pools, then
    # take n_val (fetch-all -> filter -> slice, so we still get n_val after filtering).
    fit_pool = set(harmful_extract) | set(harmful_train)
    harmful_val = [p for p in fetch(a.val_source) if p not in fit_pool][: a.n_val]

    # Scored pool MUST be disjoint from every fit/train/select pool. Several harmful
    # benchmarks (JailbreakBench especially) are partly SOURCED from AdvBench/HarmBench,
    # so filter out overlaps rather than silently leaking scored prompts into training.
    reserved = fit_pool | set(harmful_val)
    raw_eval = fetch(a.eval_source)
    harmful_eval = [p for p in raw_eval if p not in reserved]
    dropped = len(raw_eval) - len(harmful_eval)
    if not harmful_eval:
        raise SystemExit(
            f"No eval prompts left after removing overlap with extract/val/train. "
            f"'{a.eval_source}' is too small or fully contained in the reserved pools "
            f"(e.g. n_val consumed the whole pool). Lower --n-val or pick a bigger --eval-source.")

    # Alpaca: first n_harmless -> diff-in-means contrast; the next n_benign_train -> benign
    # DPO comply-pairs (disjoint slice). XSTest -> over-refusal eval only.
    alpaca = fetch("alpaca", a.n_harmless + a.n_benign_train)
    harmless = alpaca[: a.n_harmless]
    xstest = fetch("xstest")
    # Benign DPO-training prompts to counter over-refusal: extra Alpaca (general helpfulness)
    # + OR-Bench-hard (looks-dangerous-but-safe). Kept disjoint from the XSTest eval set.
    benign_train = [p for p in (alpaca[a.n_harmless:] + fetch("orbench-hard", a.n_benign_train))
                    if p not in set(xstest)]

    _write(out / "harmful_extract.txt", harmful_extract)
    _write(out / "harmful_train.txt", harmful_train)
    _write(out / "harmful_val.txt", harmful_val)
    _write(out / "harmless.txt", harmless)
    _write(out / "benign.txt", xstest)
    _write(out / "benign_train.txt", benign_train)
    _write(out / "harmful_eval.txt", harmful_eval)

    if dropped:
        why = ("reserved for the val split of the same pool" if a.val_source == a.eval_source
               else "overlapping extract/val/train")
        print(f"NOTE: removed {dropped} '{a.eval_source}' eval prompts ({why}); "
              f"kept {len(harmful_eval)} disjoint.")
    print(f"\nPools (mutually disjoint): extract={len(harmful_extract)}  "
          f"dpo_train={len(harmful_train)}  val={a.val_source}({len(harmful_val)})  "
          f"eval={a.eval_source}({len(harmful_eval)})")
    print("Next:")
    print(f"  bash scripts/run_all.sh {out}/harmful_extract.txt {out}/harmful_val.txt "
          f"{out}/harmful_train.txt {out}/harmless.txt {out}/benign.txt {out}/harmful_eval.txt")


if __name__ == "__main__":
    main()
