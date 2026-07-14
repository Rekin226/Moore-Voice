"""Quick zero-shot verification of NLLB-200 on Mooré (mos_Latn).

Run:
    uv run --with 'transformers>=4.44' --with torch python scripts/verify_nllb_baseline.py

Or:
    pip install 'transformers>=4.44' torch
    python scripts/verify_nllb_baseline.py

Loads facebook/nllb-200-distilled-600M (small, CPU-friendly) and translates a
handful of French + English sentences into Mooré (`mos_Latn`), then back.
You inspect the output as a native speaker and get an honest gut feel for how
much fine-tuning Mooré-Voice will actually need.

Uses the distilled 600M variant for speed. For real evaluation switch to
`facebook/nllb-200-3.3B` or `facebook/nllb-200-1.3B`.
"""

from __future__ import annotations

MODEL_ID = "facebook/nllb-200-distilled-600M"

SAMPLES: list[tuple[str, str, str]] = [
    ("fra_Latn", "mos_Latn", "Bonjour, comment allez-vous aujourd'hui ?"),
    ("fra_Latn", "mos_Latn", "L'eau souterraine est essentielle pour l'agriculture."),
    ("fra_Latn", "mos_Latn", "Le marché ouvre à sept heures du matin."),
    ("eng_Latn", "mos_Latn", "The children are going to school."),
    ("eng_Latn", "mos_Latn", "It will rain tomorrow morning."),
    ("eng_Latn", "mos_Latn", "Please give me a glass of water."),
]

ROUND_TRIP: list[tuple[str, str]] = [
    ("mos_Latn", "fra_Latn"),
    ("mos_Latn", "eng_Latn"),
]


def main() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading {MODEL_ID} ... (first run downloads ~2.5 GB)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    def translate(text: str, src: str, tgt: str) -> str:
        tokenizer.src_lang = src
        inputs = tokenizer(text, return_tensors="pt")
        tgt_token_id = tokenizer.convert_tokens_to_ids(tgt)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_new_tokens=128,
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print("\n=== Forward translation (X → Mooré) ===")
    forward_outputs: list[tuple[str, str, str]] = []
    for src, tgt, text in SAMPLES:
        out = translate(text, src, tgt)
        forward_outputs.append((src, text, out))
        print(f"[{src} → {tgt}] {text}")
        print(f"                → {out}\n")

    print("=== Round-trip (Mooré → X) using forward outputs ===")
    for src_orig, orig, mos_text in forward_outputs:
        for _mos, tgt in ROUND_TRIP:
            if tgt != src_orig:
                continue
            back = translate(mos_text, "mos_Latn", tgt)
            print(f"[mos_Latn → {tgt}] {mos_text}")
            print(f"                original: {orig}")
            print(f"                back:     {back}\n")

    print("Done. Native speaker: rate each Mooré output 1-5 for fluency and adequacy.")
    print("Log ratings in data/CORPORA.md under the Phase 0 checklist.")


if __name__ == "__main__":
    main()
