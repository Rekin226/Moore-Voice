"""Publish Mooré-Voice artifacts to the Hugging Face Hub.

Pushes (each optional, skipped if the local dir is missing):
  - MT LoRA adapter(s)  → Rekin226/nllb-200-<size>-moore-lora-v0
  - Whisper ASR model   → Rekin226/whisper-small-moore-v0
Each repo gets a model card with training data provenance, eval scores
(read from .logs/eval_*.json if present), and honest limitations.

Run:
    uv run --python 3.12 --with huggingface_hub python scripts/publish_hf.py \
        [--private] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS = REPO_ROOT / ".logs"

CARD_TEMPLATE = """---
language:
  - mos
  - fr
  - en
license: cc-by-nc-4.0
base_model: {base_model}
tags:
  - translation
  - moore
  - mossi
  - burkina-faso
  - low-resource
  - africa
{extra_tags}---

# {title}

{description}

Part of [Mooré-Voice](https://github.com/Rekin226/Moore-Voice) — open translation
and speech recognition for Mooré (Mòoré / Mossi, ISO 639-3 `mos`), spoken by
~8 million people in and around Burkina Faso.

## Evaluation

{eval_section}

## Training data

{data_section}

## Limitations

{limitations}

## License note

Released **CC-BY-NC-4.0** because a large share of the training text derives
from sources whose redistribution terms are research-use-only or undeclared
(see the repo's `data/CORPORA.md` / `data/AUDIO_CORPORA.md`). A fully
permissive release is planned once the corpus is rebuilt on cleared sources
(Common Voice `mos` + translatewiki + NLLB-mined).
"""


def eval_table(paths: list[Path]) -> str:
    rows = ["| Model | Direction | BLEU | chrF++ |", "|---|---|---:|---:|"]
    found = False
    for p in paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        label = "fine-tuned" if d.get("adapter") else "zero-shot base"
        for direction, s in d.get("directions", {}).items():
            rows.append(f"| {label} | `{direction}` | {s['bleu']} | {s['chrf++']} |")
            found = True
    return "\n".join(rows) if found else "_Pending — run `scripts/evaluate.py`._"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()["name"]

    jobs = []

    mt_dirs = {
        "nllb-600M-moore-lora-v0": ("facebook/nllb-200-distilled-600M",
                                    [LOGS / "eval_600M_base.json", LOGS / "eval_600M_lora.json"]),
        "nllb-3.3B-moore-lora-v0": ("facebook/nllb-200-3.3B",
                                    [LOGS / "eval_33B_base.json", LOGS / "eval_33B_lora.json"]),
    }
    for name, (base, evals) in mt_dirs.items():
        local = REPO_ROOT / "models" / name
        if not local.exists():
            print(f"[skip] {local} missing")
            continue
        card = CARD_TEMPLATE.format(
            base_model=base,
            extra_tags="  - peft\n  - lora\n",
            title=f"NLLB-200 Mooré LoRA ({name.split('-')[1]})",
            description=(f"LoRA adapter for `{base}` fine-tuned on ~205k cleaned "
                         "Mooré↔French/English sentence pairs, all four directions "
                         "(`eng_Latn↔mos_Latn`, `fra_Latn↔mos_Latn`)."),
            eval_section=eval_table(evals),
            data_section=("Curated corpus v0.1 (see repo `data/CORPORA.md`): MT560 "
                          "(Bible-register, ~89%), community instruction pairs, "
                          "NLLB-mined bitext (LASER ≥ 1.15), translatewiki. "
                          "Detokenised, LID-gated, FLORES-decontaminated. "
                          "FLORES-200 devtest held out for eval."),
            limitations=("- Register skew: mostly religious text → weaker on "
                         "administrative/technical register.\n"
                         "- Mooré orthography follows the 1976/2003 standard as used "
                         "by the source corpora; diacritic usage varies upstream.\n"
                         "- Not human-evaluated yet; BLEU/chrF++ on FLORES only."),
        )
        jobs.append((f"{user}/{name}", local, card))

    asr_local = REPO_ROOT / "models" / "whisper-small-mos-v0"
    if asr_local.exists():
        card = CARD_TEMPLATE.format(
            base_model="openai/whisper-small",
            extra_tags="  - automatic-speech-recognition\n  - whisper\n",
            title="Whisper-small Mooré ASR",
            description=("`openai/whisper-small` fine-tuned for Mooré speech→text on "
                         "~38k transcribed utterances (anchor language token: `yo`; "
                         "always pass `language='yo', task='transcribe'`)."),
            eval_section=eval_table([LOGS / "asr_whisper_ft.json",
                                     LOGS / "asr_mms_base.json"]).replace("BLEU", "WER")
                         .replace("chrF++", "CER"),
            data_section=("Community Mooré audio from Hugging Face (see repo "
                          "`data/AUDIO_CORPORA.md`): hfdjobii TTS sets + Minervus00 "
                          "collection. Audio is NOT redistributed — weights only."),
            limitations=("- Read speech dominates → spontaneous/telephone speech "
                         "will degrade.\n- Speaker diversity is limited.\n"
                         "- WER normalisation strips punctuation/case."),
        )
        jobs.append((f"{user}/whisper-small-moore-v0", asr_local, card))
    else:
        print(f"[skip] {asr_local} missing")

    for repo_id, local, card in jobs:
        has_weights = any(local.glob("*.safetensors")) or any(local.glob("*.bin"))
        if not has_weights:
            print(f"[skip] {local} has no weights (training incomplete?)")
            continue
        print(f"[publish] {repo_id}  ←  {local}")
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        (local / "README.md").write_text(card)
        api.upload_folder(repo_id=repo_id, folder_path=str(local), repo_type="model",
                          ignore_patterns=["checkpoint-*", "*.bin.tmp"])
        print(f"          → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
