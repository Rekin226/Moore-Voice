"""LoRA fine-tune of NLLB-200 on the Mooré-Voice parallel corpus.

Trains all four directions in one adapter (eng/fra → mos and mos → eng/fra);
NLLB's language tokens handle the switching. Mixed-direction batches are fine
at train time because the target language token is the first label token —
forced_bos_token_id is only needed at generation time.

Shakedown (600M, 20k subset, ~30 min on RTX 4070):
    uv run --python 3.12 --extra train python scripts/finetune_lora.py \
        --model facebook/nllb-200-distilled-600M --subset 20000 --epochs 1 \
        --output models/nllb-600M-moore-lora-shakedown

Full run (600M):
    ... --model facebook/nllb-200-distilled-600M --epochs 2 \
        --output models/nllb-600M-moore-lora-v0

Full run (3.3B, bf16 + gradient checkpointing, overnight):
    ... --model facebook/nllb-200-3.3B --epochs 1 --batch 2 --accum 16 \
        --grad-checkpoint --output models/nllb-3.3B-moore-lora-v0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "processed" / "moore_parallel_v0_1.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--subset", type=int, default=0,
                   help="train on a random subset of N rows (0 = all)")
    p.add_argument("--max-per-source", type=int, default=0,
                   help="cap rows per source to rebalance register (0 = off)")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(corpus: str, split: str, subset: int, max_per_source: int,
               seed: int) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd

    df = pd.read_parquet(corpus)
    df = df[df["split"] == split].reset_index(drop=True)
    if max_per_source:
        df = (df.groupby("source", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_per_source),
                                          random_state=seed)))
    if subset and len(df) > subset:
        df = df.sample(subset, random_state=seed)
    return df.reset_index(drop=True)


def tokenize_by_direction(df, tokenizer, max_len: int, seed: int):
    """Batch-tokenize each (src_lang, tgt_lang) group with the tokenizer's
    language state set correctly, then interleave."""
    from datasets import Dataset, concatenate_datasets

    parts = []
    for (src_lang, tgt_lang), g in df.groupby(["src_lang", "tgt_lang"]):
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        enc = tokenizer(
            list(g["src_text"]),
            text_target=list(g["tgt_text"]),
            max_length=max_len,
            truncation=True,
        )
        parts.append(Dataset.from_dict(dict(enc)))
        print(f"  tokenized {src_lang}→{tgt_lang}: {len(g):,}")
    return concatenate_datasets(parts).shuffle(seed=seed)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    print(f"[data] {args.corpus}")
    train_df = load_split(args.corpus, "train", args.subset,
                          args.max_per_source, args.seed)
    dev_df = load_split(args.corpus, "dev", 2000, 0, args.seed)
    print(f"[data] train={len(train_df):,} dev={len(dev_df):,}")
    print(train_df.groupby(["src_lang", "tgt_lang"]).size())

    print(f"[model] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("[tokenize] train")
    train_ds = tokenize_by_direction(train_df, tokenizer, args.max_len, args.seed)
    print("[tokenize] dev")
    dev_ds = tokenize_by_direction(dev_df, tokenizer, args.max_len, args.seed)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                      label_pad_token_id=-100)

    total_steps = int(len(train_ds) * args.epochs / (args.batch * args.accum))
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=max(50, total_steps // 33),  # ~3%
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
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[done] adapter saved → {args.output}")


if __name__ == "__main__":
    main()
