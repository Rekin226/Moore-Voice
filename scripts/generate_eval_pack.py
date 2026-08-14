"""Generate the blind A/B evaluation pack for human accuracy testing.

For each test sentence, produces translations from BOTH the zero-shot
NLLB-3.3B base and the fine-tuned LoRA, shuffles their A/B position with a
recorded seed, and writes data/eval_pack/eval_pack.jsonl. The Streamlit
rating app (eval_app.py) then collects blind judgments — raters never see
which system is which.

Items:
  - 20 everyday fra/eng → mos sentences (market, clinic, farm, transport…)
  - 20 held-out FLORES devtest mos → fra/eng sentences (reference included)

Run (GPU, ~15 min — loads 3.3B twice):
    uv run --python 3.12 --with 'transformers>=4.44' --with 'peft>=0.11' \
      --with torch --with sentencepiece --with protobuf --with pandas \
      --with pyarrow python scripts/generate_eval_pack.py
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = "facebook/nllb-200-3.3B"
ADAPTER = REPO_ROOT / "models" / "nllb-3.3B-moore-lora-v0"
OUT_DIR = REPO_ROOT / "data" / "eval_pack"
SEED = 42

INTO_MOS = [
    ("fra_Latn", "Le marché de Ouagadougou ouvre à sept heures du matin."),
    ("fra_Latn", "Lavez-vous les mains avant de manger."),
    ("fra_Latn", "La pluie a abîmé la route qui mène au village."),
    ("fra_Latn", "Combien coûte un sac de maïs aujourd'hui ?"),
    ("fra_Latn", "L'infirmière donne le médicament à l'enfant malade."),
    ("fra_Latn", "Mon frère répare les motos au grand marché."),
    ("fra_Latn", "Buvez beaucoup d'eau pendant la saison chaude."),
    ("fra_Latn", "La réunion du village aura lieu jeudi prochain."),
    ("fra_Latn", "Les femmes vendent des mangues au bord de la route."),
    ("fra_Latn", "L'école ferme pendant la fête."),
    ("eng_Latn", "The farmers are planting millet before the rainy season."),
    ("eng_Latn", "Please bring your vaccination card to the clinic."),
    ("eng_Latn", "The bus to Koudougou leaves at noon."),
    ("eng_Latn", "Clean drinking water keeps children healthy."),
    ("eng_Latn", "The teacher writes the lesson on the blackboard."),
    ("eng_Latn", "My grandmother tells stories in the evening."),
    ("eng_Latn", "The price of fuel went up again this month."),
    ("eng_Latn", "Wash the vegetables before cooking them."),
    ("eng_Latn", "The phone network is weak in our village."),
    ("eng_Latn", "Everyone should plant a tree this year."),
]


def load_model(with_adapter: bool):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    if with_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device).eval(), device


def translate_all(items: list[dict], with_adapter: bool, key: str) -> None:
    import torch

    tok, model, device = load_model(with_adapter)
    for it in items:
        tok.src_lang = it["src_lang"]
        enc = tok(it["src_text"], return_tensors="pt", truncation=True,
                  max_length=192).to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc, forced_bos_token_id=tok.convert_tokens_to_ids(it["tgt_lang"]),
                num_beams=5, max_new_tokens=128)
        it[key] = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        print(f"[{key}] {it['src_text'][:40]}… → {it[key][:50]}…", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    import pandas as pd

    rng = random.Random(SEED)
    items: list[dict] = []
    for src_lang, text in INTO_MOS:
        items.append({"src_lang": src_lang, "src_text": text,
                      "tgt_lang": "mos_Latn", "reference": ""})

    df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "moore_parallel_v0_1.parquet")
    ev = df[(df["split"] == "eval") & (df["src_lang"] == "mos_Latn")]
    fra = ev[ev["tgt_lang"] == "fra_Latn"].iloc[::101].head(10)
    eng = ev[ev["tgt_lang"] == "eng_Latn"].iloc[53::101].head(10)
    for part in (fra, eng):
        for _, r in part.iterrows():
            items.append({"src_lang": "mos_Latn", "src_text": r["src_text"],
                          "tgt_lang": r["tgt_lang"], "reference": r["tgt_text"]})

    # Generate with both systems (two sequential model loads to fit 12 GB).
    translate_all(items, with_adapter=False, key="base")
    translate_all(items, with_adapter=True, key="lora")

    # Blind A/B assignment.
    for i, it in enumerate(items):
        flip = rng.random() < 0.5
        it["id"] = i
        it["A"], it["B"] = (it["lora"], it["base"]) if flip else (it["base"], it["lora"])
        it["A_is"] = "lora" if flip else "base"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "eval_pack.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (OUT_DIR / "meta.json").write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL, "adapter": str(ADAPTER.name), "items": len(items),
        "seed": SEED,
    }, indent=2))
    print(f"[write] {OUT_DIR / 'eval_pack.jsonl'} ({len(items)} items)")


if __name__ == "__main__":
    main()
