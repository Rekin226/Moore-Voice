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


# Eval JSONs record the local checkpoint path; public cards want a real name.
ASR_LABELS = {
    "asr_mms_base.json": "MMS-1b-all, Meta's `mos` adapter (zero-shot)",
    "asr_whisper_ft.json": "Whisper-small fine-tuned v0",
    "asr_mms_ft.json": "**MMS-1b-all `mos` adapter fine-tuned v0**",
}


def asr_eval_table(paths: list[Path]) -> str:
    """ASR evals are flat {wer, cer, n} — not the per-direction MT shape."""
    rows = ["| Model | WER ↓ | CER ↓ | n |", "|---|---:|---:|---:|"]
    found = False
    for p in paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        label = ASR_LABELS.get(p.name, d["model"])
        rows.append(f"| {label} | {d['wer']} | {d['cer']} | {d['n']} |")
        found = True
    if not found:
        return "_Pending — run `scripts/evaluate_asr.py`._"
    return "\n".join(rows) + (
        "\n\nAll three scored on the same held-out test split with identical "
        "text normalisation (`scripts/evaluate_asr.py`)."
    )


def human_eval_section() -> str:
    """Blind A/B judgments, pooled from data/eval_pack/ratings_*.jsonl."""
    pack = REPO_ROOT / "data" / "eval_pack" / "eval_pack.jsonl"
    if not pack.exists():
        return ""
    items = {json.loads(ln)["id"]: json.loads(ln) for ln in pack.open()}
    tally: dict[str, int] = {"lora": 0, "base": 0, "tie": 0, "both_bad": 0}
    for path in sorted(pack.parent.glob("ratings_*.jsonl")):
        for ln in path.open():
            r = json.loads(ln)
            it = items.get(r["id"])
            if it is None:
                continue
            c = r["choice"]
            if c == "A is better":
                tally[it["A_is"]] += 1
            elif c == "B is better":
                tally["base" if it["A_is"] == "lora" else "lora"] += 1
            elif c == "Both equally good":
                tally["tie"] += 1
            else:
                tally["both_bad"] += 1
    n = sum(tally.values())
    if not n:
        return ""
    decided = tally["lora"] + tally["base"]
    rate = f"{100 * tally['lora'] / decided:.0f}%" if decided else "—"
    return (
        "\n\n### Native-speaker blind A/B\n\n"
        f"A native Mooré speaker compared this adapter against the zero-shot "
        f"base model on {n} unlabelled, randomly-ordered pairs:\n\n"
        f"- **Fine-tuned wins {rate} of decided pairs** "
        f"({tally['lora']}/{decided}) — statistically indistinguishable from "
        "a coin flip.\n"
        f"- {tally['tie']} pairs rated equally good, "
        f"**{tally['both_bad']} rated \"both bad\"** "
        f"({100 * tally['both_bad'] / n:.0f}% of all pairs).\n"
        "- Only `mos_Latn→eng_Latn` shows a clear human-visible gain.\n\n"
        "**Read the BLEU table above with that in mind.** The chrF++/BLEU gains "
        "are real but do not translate into quality a native speaker can "
        "perceive, except into English. Raw judgments are in the repo under "
        "`data/eval_pack/`."
    )


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
            eval_section=eval_table(evals) + human_eval_section(),
            data_section=("Curated corpus v0.1 (see repo `data/CORPORA.md`): MT560 "
                          "(Bible-register, ~89%), community instruction pairs, "
                          "NLLB-mined bitext (LASER ≥ 1.15), translatewiki. "
                          "Detokenised, LID-gated, FLORES-decontaminated. "
                          "FLORES-200 devtest held out for eval."),
            limitations=("- **A native speaker could not reliably tell this adapter "
                         "from the zero-shot base model** (50% win rate over 22 "
                         "decided pairs, sign test p = 0.58). Treat the BLEU/chrF++ "
                         "gains as a proxy that human judgment does not corroborate, "
                         "except for `mos_Latn→eng_Latn`.\n"
                         "- 30% of A/B pairs were rated \"both bad\" — for many "
                         "inputs neither model produces a usable translation.\n"
                         "- Register skew: mostly religious text (~89% Bible-register) "
                         "→ weaker on administrative/technical register. This is the "
                         "most likely cause of the above and the next thing to fix.\n"
                         "- Into-Mooré is the weak direction (`eng→mos` chrF++ is "
                         "-0.36 vs zero-shot); out-of-Mooré is where the gains are.\n"
                         "- Mooré orthography follows the 1976/2003 standard as used "
                         "by the source corpora; diacritic usage varies upstream.\n"
                         "- Human evaluation is a single rater over 40 items — enough "
                         "to say no overall gain is detectable, not enough to rank "
                         "directions confidently."),
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
            eval_section=(asr_eval_table([LOGS / "asr_whisper_ft.json",
                                          LOGS / "asr_mms_base.json",
                                          LOGS / "asr_mms_ft.json"])
                          + "\n\n> **Superseded.** The fine-tuned MMS-1b `mos` "
                            "adapter reaches WER 0.168 on the same test split — "
                            "less than half the error of this model. Prefer "
                            "[`Rekin226/mms-1b-moore-v0`](https://huggingface.co/"
                            "Rekin226/mms-1b-moore-v0) for Mooré ASR. This model "
                            "is kept for reproducibility."),
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

    mms_local = REPO_ROOT / "models" / "mms-1b-mos-v0"
    if mms_local.exists():
        card = CARD_TEMPLATE.format(
            base_model="facebook/mms-1b-all",
            extra_tags="  - automatic-speech-recognition\n  - wav2vec2\n  - mms\n",
            title="MMS-1b Mooré ASR (fine-tuned `mos` adapter)",
            description=("Meta `facebook/mms-1b-all` with its `mos` adapter "
                         "fine-tuned on ~85 h of transcribed Mooré audio. Only the "
                         "adapter layers and CTC head train (2.2M of 964M params, "
                         "0.23%); the 1B wav2vec2 base stays frozen. Warm-started "
                         "from Meta's pretrained `mos` adapter rather than "
                         "reinitialised, so training begins at the zero-shot "
                         "baseline instead of from scratch.\n\n"
                         "`adapter.mos.safetensors` (8.6 MB) is the portable "
                         "artefact — load it onto stock `mms-1b-all` instead of "
                         "pulling the full 3.9 GB checkpoint."),
            eval_section=asr_eval_table([LOGS / "asr_mms_base.json",
                                         LOGS / "asr_whisper_ft.json",
                                         LOGS / "asr_mms_ft.json"]),
            data_section=("Community Mooré audio from Hugging Face (see repo "
                          "`data/AUDIO_CORPORA.md`): hfdjobii TTS sets + Minervus00 "
                          "collection. 32,463 utterances after a 15 s duration cap "
                          "(89.8% of the corpus). Audio is NOT redistributed — "
                          "weights only."),
            limitations=("- Outputs are lowercase, unpunctuated, accent-folded: the "
                         "pretrained 39-token `mos` CTC vocab was kept, and "
                         "transcripts were normalised into it. Restoring "
                         "orthography needs a separate post-processing step.\n"
                         "- 1.0% of utterances (349 of 32,810) were dropped for "
                         "holding characters outside that vocab.\n"
                         "- Read speech dominates → spontaneous/telephone speech "
                         "will degrade.\n- Speaker diversity is limited.\n"
                         "- WER is computed after the same normalisation, so it is "
                         "not comparable to systems scored with punctuation/case."),
        )
        jobs.append((f"{user}/mms-1b-moore-v0", mms_local, card))
    else:
        print(f"[skip] {mms_local} missing")

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
