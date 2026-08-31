"""Fine-tune the MMS-1b-all `mos` adapter on the Mooré ASR corpus.

MMS keeps the 1B wav2vec2 base frozen and trains small per-language adapter
layers (~2M params) plus the CTC head. We warm-start from Meta's pretrained
`mos` adapter — already the best result on our test split (WER 31.1%
zero-shot, `docs/RESULTS_v0.md`) — instead of reinitialising it, so training
only has to improve on that baseline.

Text is normalised into the 39-token `mos` CTC vocab (lowercase, accents
folded, punctuation stripped) using the same rules as `evaluate_asr.py`.
Measured OOV against that vocab is 0.06% of characters / 1.0% of utterances;
utterances still holding an out-of-vocab character after folding are dropped.

Shakedown:
    uv run --python 3.12 --extra train --extra asr python scripts/finetune_mms.py \
        --subset 2000 --epochs 1 --output models/mms-1b-mos-shakedown

Full:
    uv run --python 3.12 --extra train --extra asr python scripts/finetune_mms.py \
        --epochs 3 --output models/mms-1b-mos-v0
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "audio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/mms-1b-all")
    p.add_argument("--lang", default="mos", help="MMS adapter / target language")
    p.add_argument("--manifest", default=str(AUDIO_DIR / "manifest.jsonl"))
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="adapters train fast")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--max-duration", type=float, default=15.0,
                   help="drop training clips longer than this (VRAM)")
    p.add_argument("--dev-size", type=int, default=200)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    return p.parse_args()


def normalize(s: str) -> str:
    """WER normalisation: NFC, lowercase, strip punctuation, squeeze spaces.

    Identical to `evaluate_asr.normalize` so training labels and the metric
    agree on what counts as a token.
    """
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s'-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fold_accents(s: str) -> str:
    """Strip combining accents that the mos vocab lacks (é→e), keeping the
    precomposed Mooré vowels (ã ẽ ĩ õ ũ) which the vocab does contain."""
    keep = set("ãẽĩõũ")
    out = []
    for ch in s:
        if ch in keep:
            out.append(ch)
            continue
        d = unicodedata.normalize("NFD", ch)
        out.append("".join(c for c in d if not unicodedata.combining(c)))
    return unicodedata.normalize("NFC", "".join(out))


def load_manifest(path: str, split: str, subset: int, seed: int,
                  max_duration: float) -> list[dict]:
    import random
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["split"] == split]
    if max_duration:
        rows = [r for r in rows if r["duration_s"] <= max_duration]
    if subset and len(rows) > subset:
        rows = random.Random(seed).sample(rows, subset)
    return rows


@dataclass
class Collator:
    processor: object

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_values": f["input_values"]} for f in features],
            return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100)
        return batch


def main() -> None:
    args = parse_args()

    import jiwer
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Dataset
    from transformers import AutoProcessor, Trainer, TrainingArguments, Wav2Vec2ForCTC

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.set_target_lang(args.lang)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        target_lang=args.lang,          # warm-start from the pretrained mos adapter
        ignore_mismatched_sizes=True,   # CTC head is resized to the mos vocab
        torch_dtype=torch.float32,      # CTC loss is unstable in bf16
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
    )

    # Train the adapter layers + CTC head only; the 1B base stays frozen.
    model.freeze_base_model()
    adapter_params = 0
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
        if param.requires_grad:
            adapter_params += param.numel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {adapter_params:,} / {total_params:,} "
          f"({100 * adapter_params / total_params:.2f}%)")

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    vocab = set(processor.tokenizer.get_vocab())  # flat after set_target_lang
    dropped = {"oov": 0, "empty": 0}

    def prepare(split: str, subset: int) -> Dataset:
        rows = load_manifest(args.manifest, split, subset, args.seed,
                             args.max_duration)
        print(f"[data] {split}: {len(rows):,} utterances "
              f"(<= {args.max_duration}s)")

        def gen():
            for r in rows:
                text = fold_accents(normalize(r["text"]))
                if not text:
                    dropped["empty"] += 1
                    continue
                if set(text.replace(" ", "")) - vocab:
                    dropped["oov"] += 1
                    continue
                audio, sr = sf.read(AUDIO_DIR / r["path"], dtype="float32")
                values = processor(
                    audio, sampling_rate=sr).input_values[0]
                labels = processor.tokenizer(text).input_ids
                # CTC needs at least one frame per label.
                if len(labels) == 0 or len(values) // 320 <= len(labels):
                    dropped["empty"] += 1
                    continue
                yield {"input_values": np.asarray(values, dtype=np.float32),
                       "labels": labels}

        return Dataset.from_generator(gen)

    train_ds = prepare("train", args.subset)
    dev_ds = prepare("dev", min(args.subset, args.dev_size) if args.subset
                     else args.dev_size)
    print(f"[data] dropped: {dropped['oov']} oov, {dropped['empty']} unusable")
    print(f"[data] usable train {len(train_ds):,} · dev {len(dev_ds):,}")

    def compute_metrics(pred):
        ids = np.argmax(pred.predictions, axis=-1)
        label_ids = np.where(pred.label_ids != -100, pred.label_ids,
                             processor.tokenizer.pad_token_id)
        hyps = processor.batch_decode(ids)
        refs = processor.tokenizer.batch_decode(label_ids, group_tokens=False)
        pairs = [(r, h) for r, h in zip(refs, hyps, strict=False) if r.strip()]
        if not pairs:
            return {"wer": 1.0}
        refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

    steps_per_epoch = max(1, len(train_ds) // (args.batch * args.accum))
    total_steps = int(steps_per_epoch * args.epochs)
    eval_every = max(50, min(500, total_steps // 6))

    train_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=max(20, total_steps // 10),
        lr_scheduler_type="linear",
        bf16=True,                      # autocast; master weights stay fp32
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_steps=eval_every,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume)

    out = Path(args.output)
    trainer.save_model(str(out))
    processor.save_pretrained(str(out))

    # Also emit the standalone adapter file — the only artefact worth publishing.
    from safetensors.torch import save_file
    adapters = {n: p.data for n, p in model.named_parameters() if "adapter" in n}
    save_file(adapters, str(out / f"adapter.{args.lang}.safetensors"))
    print(f"[done] → {out}  (adapter.{args.lang}.safetensors, "
          f"{sum(v.numel() for v in adapters.values()):,} params)")


if __name__ == "__main__":
    main()
