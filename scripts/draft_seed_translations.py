"""Machine-draft Mooré translations of the Common Voice seed corpus.

Translates data/common_voice_seed/fr_seed_v0.txt (762 French sentences) to
Mooré with the fine-tuned NLLB-3.3B LoRA, writing one draft per line to
mos_seed_v0_draft.txt (line numbers match fr_seed_v0.txt).

⚠ These are DRAFTS for native-speaker correction — Mozilla Common Voice
requires human-validated sentences; do not submit them unreviewed.

Run:
    uv run --python 3.12 --with 'transformers>=4.44' --with 'peft>=0.11' \
      --with torch --with sentencepiece --with protobuf \
      python scripts/draft_seed_translations.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "data" / "common_voice_seed"
MODEL = "facebook/nllb-200-3.3B"
ADAPTER = REPO_ROOT / "models" / "nllb-3.3B-moore-lora-v0"


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload()
    model.to(device).eval()

    lines = (SEED_DIR / "fr_seed_v0.txt").read_text().splitlines()
    tok.src_lang = "fra_Latn"
    bos = tok.convert_tokens_to_ids("mos_Latn")
    out_lines: list[str] = []
    B = 16
    for i in range(0, len(lines), B):
        chunk = lines[i:i + B]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, forced_bos_token_id=bos,
                                 num_beams=5, max_new_tokens=96)
        out_lines.extend(t.strip() for t in tok.batch_decode(out, skip_special_tokens=True))
        print(f"{i + len(chunk)}/{len(lines)}", flush=True)

    assert len(out_lines) == len(lines)
    out_path = SEED_DIR / "mos_seed_v0_draft.txt"
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
