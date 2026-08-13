"""Fine-tune Whisper on the Mooré ASR corpus (data/audio/manifest.jsonl).

Whisper has no `mos` language token, so we anchor on a fixed existing token
(default: yo / Yoruba — African, Latin-script, low collision risk) and always
pass the same language at inference. This is the standard recipe for adding
an unseen language to Whisper.

Shakedown:
    uv run ... python scripts/finetune_whisper.py --subset 2000 --epochs 1 \
        --output models/whisper-small-mos-shakedown

Full:
    uv run ... python scripts/finetune_whisper.py --epochs 3 \
        --output models/whisper-small-mos-v0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "audio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--language", default="yo", help="anchor language token")
    p.add_argument("--manifest", default=str(AUDIO_DIR / "manifest.jsonl"))
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_manifest(path: str, split: str, subset: int, seed: int):
    import random
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["split"] == split]
    if subset and len(rows) > subset:
        rows = random.Random(seed).sample(rows, subset)
    return rows


@dataclass
class Collator:
    processor: object

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    def prepare(split: str, subset: int):
        rows = load_manifest(args.manifest, split, subset, args.seed)
        print(f"[data] {split}: {len(rows):,} utterances")

        def gen():
            for r in rows:
                audio, sr = sf.read(AUDIO_DIR / r["path"], dtype="float32")
                feats = processor.feature_extractor(
                    audio, sampling_rate=sr).input_features[0]
                labels = processor.tokenizer(r["text"]).input_ids
                if len(labels) > 448:
                    continue
                yield {"input_features": feats.astype(np.float32), "labels": labels}

        return Dataset.from_generator(gen)

    train_ds = prepare("train", args.subset)
    dev_ds = prepare("dev", min(args.subset, 500) if args.subset else 500)

    total_steps = int(len(train_ds) * args.epochs / (args.batch * args.accum))
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=max(50, total_steps // 33),
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=Collator(processor),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"[done] → {args.output}")


if __name__ == "__main__":
    main()
