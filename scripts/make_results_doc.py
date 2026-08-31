"""Generate docs/RESULTS_v0.md from the eval JSONs in .logs/.

Run after scripts/evaluate.py / evaluate_asr.py:
    uv run --python 3.12 python scripts/make_results_doc.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS = REPO_ROOT / ".logs"
OUT = REPO_ROOT / "docs" / "RESULTS_v0.md"
EVAL_PACK = REPO_ROOT / "data" / "eval_pack" / "eval_pack.jsonl"

MT_EVALS = [
    ("NLLB-600M zero-shot", "eval_600M_base.json"),
    ("NLLB-600M + LoRA v0", "eval_600M_lora.json"),
    ("NLLB-3.3B zero-shot", "eval_33B_base.json"),
    ("NLLB-3.3B + LoRA v0", "eval_33B_lora.json"),
]
ASR_EVALS = [
    ("MMS-1b-all (mos adapter, zero-shot)", "asr_mms_base.json"),
    ("Whisper-small fine-tuned v0", "asr_whisper_ft.json"),
]
DIRECTIONS = ["mos_Latn→fra_Latn", "mos_Latn→eng_Latn",
              "fra_Latn→mos_Latn", "eng_Latn→mos_Latn"]


def _winner(item: dict, choice: str) -> str:
    """Map a blind A/B choice onto the system that actually won."""
    if choice == "A is better":
        return item["A_is"]
    if choice == "B is better":
        return "base" if item["A_is"] == "lora" else "lora"
    return "tie" if choice == "Both equally good" else "both_bad"


def _sign_test(wins: int, decided: int) -> float:
    """One-sided p that a 50/50 coin produces >= `wins` of `decided`."""
    if decided == 0:
        return 1.0
    return sum(comb(decided, k) for k in range(wins, decided + 1)) / 2 ** decided


def ab_section() -> list[str]:
    """Blind A/B native-speaker judgments, if any have been collected."""
    if not EVAL_PACK.exists():
        return []
    items = {}
    for line in EVAL_PACK.open():
        it = json.loads(line)
        items[it["id"]] = it

    per_rater: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    per_dir: dict[str, Counter] = defaultdict(Counter)
    for path in sorted(EVAL_PACK.parent.glob("ratings_*.jsonl")):
        for line in path.open():
            r = json.loads(line)
            it = items.get(r["id"])
            if it is None:
                continue
            w = _winner(it, r["choice"])
            overall[w] += 1
            per_rater[r["rater"]][w] += 1
            d = (f"{it['src_lang'].replace('_Latn', '')}→"
                 f"{it['tgt_lang'].replace('_Latn', '')}")
            per_dir[d][w] += 1
    if not overall:
        return []

    n = sum(overall.values())
    decided = overall["lora"] + overall["base"]
    rate = f"{100 * overall['lora'] / decided:.0f}%" if decided else "—"
    p = _sign_test(overall["lora"], decided)
    raters = ", ".join(f"{k} ({sum(v.values())})" for k, v in per_rater.items())

    lines = [
        "",
        "## Blind A/B — native-speaker judgments",
        "",
        "Fine-tuned (NLLB-3.3B + LoRA v0) vs zero-shot NLLB-3.3B, unlabelled "
        "and in random order. Collected with `eval_app.py`; raw judgments in "
        "`data/eval_pack/`.",
        "",
        f"Raters: {raters}. Items judged: {n}.",
        "",
        "| Outcome | Count |",
        "|---|---:|",
        f"| Fine-tuned better | {overall['lora']} |",
        f"| Zero-shot better | {overall['base']} |",
        f"| Equally good | {overall['tie']} |",
        f"| Both bad | {overall['both_bad']} |",
        "",
        f"**Fine-tuned win rate on decided pairs: {rate}** "
        f"({overall['lora']}/{decided}, one-sided sign test p = {p:.2f}).",
        "",
        "| Direction | Fine-tuned | Zero-shot | Tie | Both bad |",
        "|---|---:|---:|---:|---:|",
    ]
    for d, w in per_dir.items():
        lines.append(f"| {d} | {w['lora']} | {w['base']} | {w['tie']} | "
                     f"{w['both_bad']} |")

    unusable = overall["both_bad"]
    lines += [
        "",
        "### Reading these numbers",
        "",
        f"- The automatic metrics above and this human evaluation **disagree**. "
        f"NLLB-3.3B + LoRA gains +2.10 BLEU / +2.80 chrF++ on mos→fra, but a "
        f"native speaker cannot tell the two systems apart overall "
        f"(win rate {rate}, p = {p:.2f}). BLEU is a proxy; on this evidence it "
        f"is not tracking perceived quality for Mooré.",
        f"- **{unusable} of {n} pairs were judged \"both bad\"** "
        f"({100 * unusable / n:.0f}%) — on those items neither system produced "
        "a usable translation, which no win-rate can capture.",
        "- Only **mos→eng** shows a clear win (5–0, never rated worse). The "
        "other three directions slightly favour the zero-shot model — "
        "including mos→fra, where the BLEU gain is largest. The direction "
        "with the best automatic score is not the one the rater preferred.",
        "- Into-Mooré remains the weak axis, consistent with the chrF++ table "
        "above (eng→mos is −0.36 at 3.3B).",
        "",
        "**Caveat:** single rater, "
        f"{n} items, {decided} decided pairs. That is enough to say no overall "
        "improvement is detectable, and not enough to rank the directions "
        "confidently — the per-direction rows rest on ~10 judgments each.",
        "",
    ]
    return lines


def main() -> None:
    lines = [
        "# Mooré-Voice v0 results",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} by "
        "`scripts/make_results_doc.py`. Do not edit the tables by hand._",
        "",
        "## Translation — FLORES-200 devtest (1,012 sentences/direction)",
        "",
        "chrF++ is the primary metric for Mooré; BLEU shown for comparability.",
        "",
    ]

    header = "| Model | " + " | ".join(d.replace("_Latn", "") for d in DIRECTIONS) + " |"
    lines += [f"### chrF++\n\n{header}", "|---|" + "---:|" * len(DIRECTIONS)]
    rows_bleu = [f"\n### BLEU\n\n{header}", "|---|" + "---:|" * len(DIRECTIONS)]
    any_mt = False
    for label, fname in MT_EVALS:
        p = LOGS / fname
        if not p.exists():
            continue
        any_mt = True
        d = json.loads(p.read_text())["directions"]
        lines.append("| " + label + " | " + " | ".join(
            str(d.get(k, {}).get("chrf++", "—")) for k in DIRECTIONS) + " |")
        rows_bleu.append("| " + label + " | " + " | ".join(
            str(d.get(k, {}).get("bleu", "—")) for k in DIRECTIONS) + " |")
    lines += rows_bleu if any_mt else ["", "_No MT evals found in .logs/._"]
    lines += ab_section()

    lines += ["", "## Speech recognition — held-out test split", "",
              "| Model | WER ↓ | CER ↓ | n |", "|---|---:|---:|---:|"]
    for label, fname in ASR_EVALS:
        p = LOGS / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        lines.append(f"| {label} | {d['wer']} | {d['cer']} | {d['n']} |")

    # Qualitative sample table from the best available MT eval.
    for _, fname in reversed(MT_EVALS):
        p = LOGS / fname
        if p.exists():
            d = json.loads(p.read_text())
            lines += ["", "## Samples (fine-tuned)" if "lora" in fname
                      else "## Samples (zero-shot)", ""]
            for direction, s in d["directions"].items():
                for ex in s.get("sample", [])[:2]:
                    lines += [f"**{direction}**",
                              f"- src: {ex['src']}",
                              f"- hyp: {ex['hyp']}",
                              f"- ref: {ex['ref']}", ""]
            break

    lines += [
        "## Native-speaker rating sheet (to fill)",
        "",
        "Rate each fine-tuned output 1–5 for Fluency (natural Mooré?) and",
        "Adequacy (meaning preserved?). Draw 20 sentences from the samples",
        "above plus everyday domains (market, clinic, agriculture).",
        "",
        "| # | Direction | Source | Model output | Fluency | Adequacy |",
        "|---|---|---|---|---|---|",
        "| 1 |  |  |  |  |  |",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
