"""Re-run the Phase 0 baseline with facebook/nllb-200-3.3B.

Same 6 sentences + round-trip as verify_nllb_baseline.py, but on the larger
NLLB variant. Answers: was the small model the ceiling, or is Mooré itself
still weak in NLLB-200 regardless of size?

CPU inference on a 3.3B model is slow (~30-60s per translation). Total ~10-15
min of inference alone, plus ~10 min to download the ~6 GB checkpoint.
"""

from __future__ import annotations

MODEL_ID = "facebook/nllb-200-3.3B"

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

    print(f"Loading {MODEL_ID} ... (first run downloads ~6 GB)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()

    def translate(text: str, src: str, tgt: str) -> str:
        tokenizer.src_lang = src
        inputs = tokenizer(text, return_tensors="pt")
        tgt_token_id = tokenizer.convert_tokens_to_ids(tgt)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_new_tokens=128,
            num_beams=4,
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print("\n=== Forward translation (X → Mooré) ===")
    forward_outputs: list[tuple[str, str, str]] = []
    for src, tgt, text in SAMPLES:
        out = translate(text, src, tgt)
        forward_outputs.append((src, text, out))
        print(f"[{src} → {tgt}] {text}")
        print(f"                → {out}\n", flush=True)

    print("=== Round-trip (Mooré → X) using forward outputs ===")
    for src_orig, orig, mos_text in forward_outputs:
        for _mos, tgt in ROUND_TRIP:
            if tgt != src_orig:
                continue
            back = translate(mos_text, "mos_Latn", tgt)
            print(f"[mos_Latn → {tgt}] {mos_text}")
            print(f"                original: {orig}")
            print(f"                back:     {back}\n", flush=True)

    print("Done. Compare to distilled-600M outputs.")


if __name__ == "__main__":
    main()
