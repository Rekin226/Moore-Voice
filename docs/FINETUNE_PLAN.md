# NLLB-200-3.3B → Mooré LoRA fine-tuning plan

Plan doc, not code. Concrete enough to execute; not so rigid that we can't change it once the ratings are in and the sawadogosalif access is (hopefully) granted.

## Objective

Take `facebook/nllb-200-3.3B` (already the best public open baseline for Mooré) and specialise it on curated Mooré parallel data so that:

- Fluency and adequacy on the 6-sentence native-rated benchmark **improve** vs zero-shot 3.3B
- FLORES-200 devtest BLEU on `eng_Latn → mos_Latn` and `fra_Latn → mos_Latn` **improve** vs zero-shot 3.3B
- Model size stays deployable on a single consumer GPU (LoRA adapter, not full weights)

## Non-goals for v0

- No Mooré → French/English (we can add later — start with the harder direction)
- No ASR fine-tune yet (separate track once translation baseline is in place)
- No new corpus creation (that's Common Voice 750-seed and future work)

## Corpus assembly

Sources, after deep inspection:

| Source | Est. clean pairs | Domain | Notes |
|---|---:|---|---|
| `michsethowusu/english-moore_sentence-pairs_mt560` | 186,434 | Bible-heavy En↔Mos | Well-aligned, clean. |
| `hfdjobii/mistral-moore-dataset-v2` (extract) | ~129,000 | Short Fr/En↔Mos + Mooré-Fr dictionary | Strip the DJOBI TOTO system prompt, keep the translation pair. |
| `madoss/nllb-mos-raw` filtered `laser_score >= 1.15` | ~8,100 | Religious/Bible En↔Mos | Spot-check for LID errors during processing. |
| translatewiki `fr↔mos` + `en↔mos` (OPUS) | ~5,900 + ~5,900 | Wikipedia UI (secular) | Diversifies away from religious skew. |
| `sawadogosalif/MooreFRCollections` | ? | Fr↔Mos, likely mixed | **Pending access request.** |
| FLORES-200 devtest | 1,012 | Diverse (Wikimedia) | **Held out — evaluation only.** |

Working total pre-dedup: ~335,000 pairs. Expect ~250k–280k after de-duplication and format normalisation.

### Processing pipeline

1. Load each source, cast to a common schema: `{"src_lang", "src_text", "tgt_lang", "tgt_text", "source"}`.
2. For `hfdjobii/mistral-moore-dataset-v2`:
   - Regex-extract the actual instruction from `[INST] ...system prompt... {instruction} [/INST] {answer}</s>`.
   - Detect direction from the instruction: `"Give the mooré translation of: X"` → `X` is source (auto-detect Fr/En), `answer` is Mooré.
   - Skip rows where extraction fails.
3. For `madoss/nllb-mos-raw`:
   - Filter `laser_score >= 1.15`.
   - Drop rows whose `mos_Latn` fails a lightweight Mooré-orthography check (must contain at least one of `ɛ ɩ ʋ ɔ ẽ ã ĩ õ ũ` or common Mooré function words; the deep-inspect script has a working heuristic).
4. For all sources: strip HTML/URLs, normalise whitespace, drop rows where either side is empty or > 200 tokens.
5. Deduplicate on `(src_text, tgt_text)` after casefolding + whitespace-normalising the source side.
6. Hold out FLORES-200 devtest as `eval` split. Random 5% of the rest as `dev`; remainder as `train`.
7. Write parquet: `data/processed/moore_parallel_v0.parquet`. Log per-source counts to `data/processed/manifest.json`.

## Fine-tune configuration

Method: **LoRA** on NLLB-200-3.3B via Hugging Face `peft`.

Rationale: full fine-tune of 3.3B is ~13 GB fp16 + optimizer states → needs 40 GB+ VRAM. LoRA on attention `q,v` (rank 16) is ~40 MB adapter, trainable on a single 24 GB GPU with room to spare.

Baseline hyperparameters (adjust after first run):

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # NLLB uses these names
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

train_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # effective batch 32
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,                       # or bf16 if the GPU supports it
    logging_steps=50,
    eval_steps=1000,
    save_steps=1000,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
)
```

Directions trained:
- `eng_Latn → mos_Latn`
- `fra_Latn → mos_Latn`

Both directions in the same LoRA — cheaper and NLLB's language tokens handle switching.

## Compute — where to actually run this

Your Azure resources per memory: `AiUpscale` (eastus2, AiNextLevel RG) and `rekin226` (graybox RG). Options in rough cost order:

1. **Google Colab T4 (free)** — 15 GB VRAM. Enough for 3.3B in fp16 with LoRA rank 16, batch 2. Slow (~4-6 hours per epoch on 280k pairs), but zero cost. Good for the first shakedown run.
2. **Colab Pro / A100** — ~$10 for a full training run. Much faster (~45 min per epoch).
3. **Azure `Standard_NC24ads_A100_v4`** — 1× A100 80 GB. ~$3/hr on-demand. Full run in ~2 hours, ~$6. Use if Colab hits quota or we want reproducibility on a fixed instance.
4. **Azure Machine Learning studio** with the same VM — nicer job tracking, but same underlying cost.

Recommendation: **start on Colab T4 with a 20k-pair subset** to validate the pipeline end-to-end (loss decreasing, eval running, checkpoint saving) in ~30 min. Then re-run full corpus on Colab A100 or Azure A100 for the real training.

## Evaluation

Two axes:

1. **Automatic (BLEU + chrF) on FLORES-200 devtest** (1,012 pairs). Report before/after, both directions. `sacrebleu` for both metrics.
2. **Human (native speaker) rating** on the 6 baseline sentences already used + a new 20-sentence hold-out set drawn from an unseen domain (weather forecast headlines, agricultural extension bulletins). Fluency 1-5, Adequacy 1-5, per direction.

Success criteria for v0:
- BLEU / chrF on FLORES: **at least +2 BLEU** over zero-shot 3.3B on both directions
- Native rating: **≥+0.5** average on both Fluency and Adequacy vs zero-shot 3.3B
- No regression on any single sentence in the 6-sentence baseline (i.e. adapter didn't over-fit to religious register and forget everyday Mooré)

## Deliverables

- `scripts/build_corpus.py` — corpus assembly per the pipeline above
- `scripts/finetune_lora.py` — training loop
- `scripts/evaluate.py` — BLEU/chrF on FLORES + human-rating table generator
- `data/processed/moore_parallel_v0.parquet` + `manifest.json`
- `models/nllb-mos-lora-v0/` — LoRA adapter checkpoint (`.safetensors`)
- `docs/RESULTS_v0.md` — before/after table + native-speaker ratings
- HuggingFace release: `Rekin226/nllb-200-3.3B-moore-lora-v0` (public, MIT + attribution to source datasets)

## Timeline (optimistic)

| Day | Milestone |
|---|---|
| Today | Native ratings collected (this turn) |
| +1 | Corpus assembly script + first parquet built |
| +2 | Colab T4 shakedown run on 20k subset, pipeline validated |
| +3 | Full training on Colab A100 or Azure A100 |
| +4 | Evaluation + docs, HF release |
| +5 | Post write-up, cite in Mooré-Voice README |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Religious-register skew makes the model bad at everyday Mooré | Reserve translatewiki subset + native hand-checked eval set; if regression on everyday sentences, upweight secular sources or curriculum-train (religious first, then secular). |
| `sawadogosalif` access denied or slow | Proceed without; add later as data v0.1. |
| Colab session drops mid-training | Save checkpoints every 500 steps; use `hf_hub_download` + resume. |
| LID errors in `madoss/nllb-mos-raw` slip through | Sample 100 rows post-filter, native-verify; tighten threshold to 1.20 if hit rate < 12/15. |
| BLEU improves but human ratings don't | BLEU is a proxy; trust the human ratings. If they diverge, iterate on the corpus mix, not the hyperparameters. |

## Not in this plan

- ASR (separate track; needs its own audio-corpus assembly and Whisper/Parakeet fine-tune plan)
- Mooré → Fr/En (the reverse direction — add in v0.1)
- Common Voice 750-seed corpus contribution (parallel track, doesn't block v0 fine-tune)
- Full-weight fine-tune (only if LoRA plateaus)
