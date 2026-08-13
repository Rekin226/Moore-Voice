"""Evaluate NLLB (base or base+LoRA) on the FLORES-200 devtest eval split.

Reports BLEU and chrF (sacrebleu) for all four directions:
    eng_Latn→mos_Latn, fra_Latn→mos_Latn, mos_Latn→eng_Latn, mos_Latn→fra_Latn

chrF is the primary metric for Mooré (BLEU under-tokenizes low-resource
orthographies); both are logged.

Zero-shot baseline:
    uv run --python 3.12 --extra train python scripts/evaluate.py \
        --model facebook/nllb-200-distilled-600M --out .logs/eval_600M_base.json

Fine-tuned:
    ... --adapter models/nllb-600M-moore-lora-v0 --out .logs/eval_600M_lora.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "processed" / "moore_parallel_v0_1.parquet"

DIRECTIONS = [
    ("eng_Latn", "mos_Latn"),
    ("fra_Latn", "mos_Latn"),
    ("mos_Latn", "eng_Latn"),
    ("mos_Latn", "fra_Latn"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--adapter", default=None)
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--limit", type=int, default=0, help="cap sentences per direction")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    import sacrebleu
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] {args.model} adapter={args.adapter} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.to(device).eval()

    df = pd.read_parquet(args.corpus)
    df = df[df["split"] == "eval"]
    assert len(df), "eval split is empty — rebuild the corpus first"

    results: dict = {
        "model": args.model,
        "adapter": args.adapter,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "beams": args.beams,
        "directions": {},
    }

    for src_lang, tgt_lang in DIRECTIONS:
        d = df[(df["src_lang"] == src_lang) & (df["tgt_lang"] == tgt_lang)]
        if args.limit:
            d = d.head(args.limit)
        srcs = list(d["src_text"])
        refs = list(d["tgt_text"])
        if not srcs:
            print(f"[skip] {src_lang}→{tgt_lang}: no eval rows")
            continue

        tokenizer.src_lang = src_lang
        bos = tokenizer.convert_tokens_to_ids(tgt_lang)
        hyps: list[str] = []
        for i in range(0, len(srcs), args.batch):
            chunk = srcs[i:i + args.batch]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=256).to(device)
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    forced_bos_token_id=bos,
                    num_beams=args.beams,
                    max_new_tokens=args.max_new_tokens,
                )
            hyps.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            if (i // args.batch) % 10 == 0:
                print(f"  {src_lang}→{tgt_lang}: {i + len(chunk)}/{len(srcs)}", flush=True)

        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++
        results["directions"][f"{src_lang}→{tgt_lang}"] = {
            "n": len(srcs),
            "bleu": round(bleu.score, 2),
            "chrf++": round(chrf.score, 2),
            "sample": [{"src": srcs[j], "ref": refs[j], "hyp": hyps[j]}
                       for j in range(min(3, len(srcs)))],
        }
        print(f"[score] {src_lang}→{tgt_lang}  BLEU={bleu.score:.2f}  chrF++={chrf.score:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
