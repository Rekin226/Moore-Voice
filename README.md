# Mooré-Voice

**Open translation and speech recognition for Mooré (Mòoré / Mossi, ISO 639-3 `mos`)** — the language of the Mossi people of Burkina Faso, spoken by ~8 million people across Burkina Faso, Côte d'Ivoire, Togo, Ghana, and Mali.

Built on Meta NLLB-200, Meta Omnilingual ASR, and NVIDIA NeMo. Corpus contributions upstream to Common Voice, Lanfrica, and NeMo.

_Repo name on GitHub is `Moore-Voice` (no accent — GitHub restriction). Human name is `Mooré-Voice`._

---

## Status (2026-08)

| Track | Status |
|---|---|
| Public Mooré dataset audit (HF + OPUS + Common Voice + Wikipedia) | ✅ done — catalogued in `data/CORPORA.audit.json` |
| Curated parallel corpus **v0.1** | ✅ **205,271 clean pairs / 414,590 direction-rows** — detokenised, LID-gated on every Mooré side, fragment-filtered, pair-level splits, all 4 directions (`data/processed/`, rebuild with `scripts/build_corpus.py`) |
| FLORES-200 eval split | ✅ **1,012 devtest sentences × 4 directions**, fetched from Meta's public mirror (no gating), train/dev decontaminated against it on both sides |
| NLLB LoRA fine-tune | ✅ pipeline validated (shakedown) → full 600M + 3.3B runs on local RTX 4070; adapters in `models/` |
| Automatic evaluation (BLEU / chrF++) | ✅ `scripts/evaluate.py`, before/after in `docs/RESULTS_v0.md` |
| ASR corpus (~38k transcribed utterances) | ✅ `scripts/build_asr_corpus.py` → `data/audio/` (see `data/AUDIO_CORPORA.md`) |
| ASR fine-tune (Whisper-small) | ✅ trained — WER 34.1% / CER 11.4% on held-out test; MMS-1b zero-shot baseline WER 31.1% (`docs/RESULTS_v0.md`) |
| Published models | ✅ [`Rekin226/nllb-3.3B-moore-lora-v0`](https://huggingface.co/Rekin226/nllb-3.3B-moore-lora-v0) · [`Rekin226/nllb-600M-moore-lora-v0`](https://huggingface.co/Rekin226/nllb-600M-moore-lora-v0) · [`Rekin226/whisper-small-moore-v0`](https://huggingface.co/Rekin226/whisper-small-moore-v0) |
| Demo app (translate + speech→text→translate) | ✅ `app.py` (Gradio, local or HF Space) |
| Common Voice `mos` locale — 750 seed sentences | ✅ **762 French sentences** ready in `data/common_voice_seed/fr_seed_v0.txt` |
| Mooré translations of the seed corpus | ⏳ awaiting native-speaker work → `mos_seed_v0.txt` — **the one step no machine can do** |
| Native-speaker rating of model outputs | ⏳ rating sheet in `docs/RESULTS_v0.md` |

## Why

The models exist. **NLLB-200 already supports Mooré as `mos_Latn`**, and Meta's **Omnilingual ASR** (Nov 2025) covers 1,600+ languages including African ones. The bottleneck for real-world Mooré language technology is *not* compute or architecture — it is a curated, licensed, community-validated corpus of parallel text and transcribed audio. This project builds that corpus and the fine-tuning recipes on top.

## Scope (v0)

- **Translation:** French ↔ Mooré, English ↔ Mooré. Fine-tune NLLB-200 on curated Mooré parallel data.
- **Transcription (ASR):** Mooré → text. Fine-tune NVIDIA NeMo Parakeet-TDT and Meta Omnilingual ASR against a Mooré audio corpus.
- **Corpus:** Community-collected + Bible/JW300 + AFRIDOC-MT + government / NGO sources. All redistributable licenses only (see `data/CORPORA.md` for policy).

Out of scope for v0: text-to-speech (Mooré TTS is a v1 target), other Voltaic languages (Fulfulde, Dagbani, Dyula).

## Stack

| Layer | Tool |
|---|---|
| Translation baseline | Meta [NLLB-200](https://ai.meta.com/blog/nllb-200-high-quality-machine-translation/) (`mos_Latn`) |
| Translation fine-tune | Hugging Face `transformers` + `peft` LoRA on NLLB-200-3.3B |
| ASR baseline | Meta [Omnilingual ASR](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/), NVIDIA [Parakeet-TDT](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/) / [Canary-1B-v2](https://arxiv.org/pdf/2509.14128) |
| ASR fine-tune | NVIDIA NeMo ASR + NeMo Speech Data Processor |
| Evaluation | FLORES-200 devtest + native-speaker held-out set |
| Compute | Colab T4 (free) → Colab A100 or Azure A100 for production runs |

## Repo layout

```
Mooré-Voice/
├── README.md                                     ← this file
├── LICENSE                                       ← MIT
├── moore_voice/                                  ← Python package (skeleton)
├── docs/FINETUNE_PLAN.md                         ← executable v0 plan
├── scripts/
│   ├── verify_nllb_baseline.py                   ← Phase-0 zero-shot test (NLLB-200-600M)
│   ├── verify_nllb_3b3.py                        ← Phase-0 zero-shot test (NLLB-200-3.3B)
│   ├── audit_existing_corpora.py                 ← HF / OPUS / Common Voice / Wikipedia audit
│   ├── inspect_moore_datasets.py                 ← surface-level dataset preview
│   ├── deep_inspect_datasets.py                  ← focused mistral-v2 + NLLB spot-check
│   └── build_corpus.py                           ← Phase-1 corpus assembly pipeline
└── data/
    ├── CORPORA.md                                ← corpus inventory + licensing policy
    ├── CORPORA.audit.json                        ← audit snapshot (2026-07)
    ├── processed/                                ← gitignored — parquet + manifest
    │   ├── moore_parallel_v0.parquet             ← 210,455 clean pairs
    │   └── manifest.json                         ← per-source / per-direction counts
    └── common_voice_seed/
        ├── README.md                             ← Mozilla submission workflow
        ├── fr_seed_v0.txt                        ← 762 French seed sentences (CC0)
        └── mos_seed_v0.txt                       ← ⏳ Mooré translations (WIP)
```

## Quickstart

```bash
# 1. Rebuild the parallel corpus (downloads pinned public sources + FLORES-200)
uv run --python 3.12 --with opustools python scripts/fetch_translatewiki_fr.py
uv run --python 3.12 --with 'datasets>=2.20' --with 'pandas>=2.0' \
  --with 'pyarrow>=15' --with huggingface_hub python scripts/build_corpus.py

# 2. Fine-tune NLLB (LoRA, fits a 12 GB consumer GPU)
uv run --python 3.12 --with 'transformers>=4.44' --with 'peft>=0.11' \
  --with torch --with sentencepiece --with protobuf --with 'accelerate>=0.30' \
  --with 'datasets>=2.20' --with pandas --with pyarrow \
  python scripts/finetune_lora.py --model facebook/nllb-200-distilled-600M \
  --epochs 2 --batch 8 --accum 4 --output models/nllb-600M-moore-lora-v0

# 3. Evaluate on FLORES-200 devtest (all 4 directions, BLEU + chrF++)
uv run ... python scripts/evaluate.py --out .logs/eval_base.json                # zero-shot
uv run ... python scripts/evaluate.py --adapter models/nllb-600M-moore-lora-v0 \
  --out .logs/eval_lora.json                                                    # fine-tuned

# 4. ASR corpus + Whisper fine-tune
uv run ... python scripts/build_asr_corpus.py
uv run ... python scripts/finetune_whisper.py --epochs 3 --output models/whisper-small-mos-v0

# 5. Demo (translation + speech-to-text)
uv run --python 3.12 --with gradio --with 'transformers>=4.44' --with 'peft>=0.11' \
  --with torch --with sentencepiece --with protobuf --with soundfile --with librosa \
  python app.py
```

Unit tests: `uv run --python 3.12 --with pytest --with pandas --with pyarrow -m pytest`.

## How to contribute

- **Translate seed sentences** for Common Voice `mos` locale — see `data/common_voice_seed/README.md`.
- **Report a bad Mooré row** in the curated corpus — open an issue with the row index or the exact source/target text; we tighten the LID heuristic in `build_corpus.py`.
- **Add a source** — put it in `data/CORPORA.md` with license, size, redistributability, and either a loader in `build_corpus.py` or a pointer script if non-redistributable.

## Roadmap

### Phase 0 — Baseline verification ✅
- [x] Run NLLB-200 zero-shot on native-speaker verified sentences
- [x] Audit existing Mooré datasets on HF, OPUS, Common Voice, Wikipedia

### Phase 1 — Corpus ✅
- [x] Assemble parallel Fr/En↔Mooré text, cleaned + LID-gated (**205k pairs, 4 directions**)
- [x] FLORES-200 devtest as held-out eval (public Meta mirror; train/dev decontaminated)
- [x] Draft 750-sentence seed corpus for Common Voice `mos` (**762 sentences done**)
- [ ] Translate seed corpus into Mooré and submit to Common Voice ← **needs a native speaker**

### Phase 2 — Translation fine-tune ✅ (local RTX 4070)
- [x] LoRA fine-tune of NLLB-200-600M on the curated corpus, all 4 directions
- [x] LoRA fine-tune of NLLB-200-3.3B (overnight run)
- [x] BLEU/chrF++ before/after on FLORES devtest → `docs/RESULTS_v0.md`
- [ ] Native-speaker rating of fine-tuned outputs (sheet in RESULTS_v0.md)

### Phase 3 — ASR ✅ (v0)
- [x] Assemble 37,654-utterance / 85 h transcribed Mooré audio corpus (`data/AUDIO_CORPORA.md`)
- [x] Fine-tune Whisper-small for Mooré speech→text (WER 34.1% test)
- [x] MMS-1b-all `mos` zero-shot baseline (WER 31.1% — currently the stronger engine; `MOORE_ASR_ENGINE=mms` in the demo)
- [ ] Fine-tune the MMS `mos` adapter on our corpus (expected to beat both)
- [ ] Access to gated CITADEL-BF / goaicorp audio (~2× more data)

### Phase 4 — Release + Upstream
- [x] Gradio demo app (translate + speech→text→translate)
- [ ] Publish adapters + demo Space under `Rekin226/*`
- [ ] Common Voice `mos` unlocked (after seed translation)
- [ ] Publish preprint (EMNLP or LREC target)

## Data licensing policy

Only **redistributable** sources are accepted into the packaged corpus. Non-redistributable sources (e.g. paywalled parallel texts, private community recordings without consent) are referenced by pointer scripts only. Every corpus entry in `data/CORPORA.md` must list: source, license, size, collection method, and consent status.

## References

- Meta [NLLB-200 paper](https://arxiv.org/pdf/2207.04672)
- Meta [Omnilingual ASR](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/) — 1600+ language ASR
- [Canary-1B-v2 & Parakeet-TDT-0.6B-v3](https://arxiv.org/pdf/2509.14128)
- [Dealing with the Hard Facts of Low-Resource African NLP](https://arxiv.org/pdf/2511.18557) — Nov 2025 survey
- [AFRIDOC-MT](https://arxiv.org/pdf/2501.06374) — document-level African MT corpus

## Related projects (same author)

- [HydroPhysicsAI](https://github.com/Rekin226/HydroPhysicsAI) — physics-ML for groundwater
- Groundwater and Mooré language work overlap in the "digital public goods for Burkina Faso" theme.

## License

MIT (see `LICENSE`). Data licensing is per-corpus; see `data/CORPORA.md`.
