"""Generate docs/RESULTS_v0.md from the eval JSONs in .logs/.

Run after scripts/evaluate.py / evaluate_asr.py:
    uv run --python 3.12 python scripts/make_results_doc.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS = REPO_ROOT / ".logs"
OUT = REPO_ROOT / "docs" / "RESULTS_v0.md"

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
