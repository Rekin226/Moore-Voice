"""Evaluate ASR models on the Mooré test split (WER / CER).

Supports:
  - fine-tuned or stock Whisper: --whisper <model-or-dir> [--language yo]
  - Meta MMS-1b-all with its mos adapter (zero-shot baseline): --mms

Run:
    uv run ... python scripts/evaluate_asr.py --mms --out .logs/asr_mms_base.json
    uv run ... python scripts/evaluate_asr.py --whisper models/whisper-small-mos-v0 \
        --out .logs/asr_whisper_ft.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "audio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--whisper", default=None)
    p.add_argument("--language", default="yo")
    p.add_argument("--mms", action="store_true")
    p.add_argument("--manifest", default=str(AUDIO_DIR / "manifest.jsonl"))
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", required=True)
    return p.parse_args()


def normalize(s: str) -> str:
    """WER normalisation: NFC, lowercase, strip punctuation, squeeze spaces."""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s'-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    args = parse_args()
    assert args.whisper or args.mms, "pick --whisper <model> or --mms"

    import jiwer
    import soundfile as sf
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(line) for line in open(args.manifest)]
    rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} utterances from split={args.split}")

    hyps: list[str] = []
    refs = [r["text"] for r in rows]

    if args.mms:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        processor.tokenizer.set_target_lang("mos")
        model.load_adapter("mos")
        label = f"{model_id}(mos adapter)"
        for i, r in enumerate(rows):
            audio, sr = sf.read(AUDIO_DIR / r["path"], dtype="float32")
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)[0]
            hyps.append(processor.decode(ids))
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    else:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        processor = WhisperProcessor.from_pretrained(
            args.whisper, language=args.language, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.whisper, torch_dtype=torch.bfloat16).to(device).eval()
        label = args.whisper
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            audios = [sf.read(AUDIO_DIR / r["path"], dtype="float32")[0] for r in chunk]
            inputs = processor(audios, sampling_rate=16000,
                               return_tensors="pt").to(device, torch.bfloat16)
            with torch.inference_mode():
                out = model.generate(inputs.input_features,
                                     language=args.language, task="transcribe",
                                     max_new_tokens=224)
            hyps.extend(processor.batch_decode(out, skip_special_tokens=True))
            if (i // args.batch) % 10 == 0:
                print(f"  {i + len(chunk)}/{len(rows)}", flush=True)

    nrefs = [normalize(r) for r in refs]
    nhyps = [normalize(h) for h in hyps]
    wer = jiwer.wer(nrefs, nhyps)
    cer = jiwer.cer(nrefs, nhyps)
    print(f"[score] {label}  WER={wer:.3f}  CER={cer:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": label,
        "split": args.split,
        "n": len(rows),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": [{"ref": refs[i], "hyp": hyps[i]} for i in range(min(5, len(rows)))],
    }, indent=2, ensure_ascii=False))
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
