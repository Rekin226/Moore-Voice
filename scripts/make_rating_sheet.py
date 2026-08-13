"""Generate docs/RATING_SHEET_v0.md — native-speaker evaluation sheet.

20 items, all produced by the fine-tuned NLLB-3.3B LoRA:
  - 10 everyday French/English sentences → Mooré (rate the Mooré)
  - 10 held-out FLORES devtest Mooré sentences → French/English
    (reference translation shown for comparison)

Fluency: 1 (broken) … 5 (natural). Adequacy: 1 (wrong meaning) … 5 (exact).

Run (GPU recommended):
    uv run --python 3.12 --with 'transformers>=4.44' --with 'peft>=0.11' \
      --with torch --with sentencepiece --with protobuf --with pandas \
      --with pyarrow python scripts/make_rating_sheet.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = "facebook/nllb-200-3.3B"
ADAPTER = REPO_ROOT / "models" / "nllb-3.3B-moore-lora-v0"
OUT = REPO_ROOT / "docs" / "RATING_SHEET_v0.md"

# Everyday register the corpus is weakest in — market, clinic, agriculture,
# weather, transport, admin. Deliberately NOT religious.
INTO_MOS = [
    ("fra_Latn", "Le marché de Ouagadougou ouvre à sept heures du matin."),
    ("fra_Latn", "Lavez-vous les mains avant de manger."),
    ("fra_Latn", "La pluie a abîmé la route qui mène au village."),
    ("fra_Latn", "Combien coûte un sac de maïs aujourd'hui ?"),
    ("fra_Latn", "L'infirmière donne le médicament à l'enfant malade."),
    ("eng_Latn", "The farmers are planting millet before the rainy season."),
    ("eng_Latn", "Please bring your vaccination card to the clinic."),
    ("eng_Latn", "The bus to Koudougou leaves at noon."),
    ("eng_Latn", "Clean drinking water keeps children healthy."),
    ("eng_Latn", "The teacher writes the lesson on the blackboard."),
]


def main() -> None:
    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
    model.to(device).eval()

    def tr(text: str, src: str, tgt: str) -> str:
        tok.src_lang = src
        enc = tok(text, return_tensors="pt", truncation=True, max_length=192).to(device)
        with torch.inference_mode():
            out = model.generate(**enc,
                                 forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                 num_beams=5, max_new_tokens=128)
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    rows = []
    for src_lang, text in INTO_MOS:
        hyp = tr(text, src_lang, "mos_Latn")
        rows.append((f"{src_lang[:3]}→mos", text, hyp, ""))
        print(f"[{src_lang[:3]}→mos] {hyp}")

    # 10 held-out FLORES mos sentences → 5 fra + 5 eng, evenly spaced sample.
    df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "moore_parallel_v0_1.parquet")
    ev = df[(df["split"] == "eval") & (df["src_lang"] == "mos_Latn")]
    fra = ev[ev["tgt_lang"] == "fra_Latn"].iloc[::202].head(5)
    eng = ev[ev["tgt_lang"] == "eng_Latn"].iloc[101::202].head(5)
    for part, tgt in ((fra, "fra_Latn"), (eng, "eng_Latn")):
        for _, r in part.iterrows():
            hyp = tr(r["src_text"], "mos_Latn", tgt)
            rows.append((f"mos→{tgt[:3]}", r["src_text"], hyp, r["tgt_text"]))
            print(f"[mos→{tgt[:3]}] {hyp}")

    lines = [
        "# Native-speaker rating sheet — v0 (NLLB-3.3B + LoRA)",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')} by "
        "`scripts/make_rating_sheet.py`. Fill Fluency and Adequacy (1–5)._",
        "",
        "- **Fluency**: is the output natural, well-formed language? 1 = broken, 5 = natural.",
        "- **Adequacy**: is the meaning of the source preserved? 1 = wrong, 5 = exact.",
        "",
        "| # | Direction | Source | Model output | Reference (if any) | Fluency | Adequacy |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (d, src, hyp, ref) in enumerate(rows, 1):
        esc = lambda s: s.replace("|", "\\|")  # noqa: E731
        lines.append(f"| {i} | {d} | {esc(src)} | {esc(hyp)} | {esc(ref)} |  |  |")
    lines += [
        "",
        "**Averages:** Fluency ___ / 5 · Adequacy ___ / 5",
        "",
        "Success criterion (docs/FINETUNE_PLAN.md): ≥ +0.5 vs the zero-shot",
        "baseline on both axes. Rate the same sentences with the base model",
        "(`MOORE_MT_ADAPTER= python app.py`) if a side-by-side is wanted.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
