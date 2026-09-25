# Mooré-Voice

**Open translation and speech recognition for Mooré (Mòoré / Mossi, ISO 639-3 `mos`)** — the language of the Mossi people of Burkina Faso, spoken by ~8 million people across Burkina Faso, Côte d'Ivoire, Togo, Ghana, and Mali.

**Try it:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rekin226/Moore-Voice/blob/main/notebooks/moore_voice_demo.ipynb) — free GPU, prints a public demo link anyone can open on a phone.

Built on Meta NLLB-200, Meta Omnilingual ASR, and NVIDIA NeMo. Corpus contributions upstream to Common Voice, Lanfrica, and NeMo.

_Repo name on GitHub is `Moore-Voice` (no accent — GitHub restriction). Human name is `Mooré-Voice`._

---

## Status (2026-09)

**Headline, honestly stated:** ASR works — the fine-tuned MMS-1b `mos` adapter reaches **WER 16.8%**, less than half the error of anything else we tried. **Translation does not yet.** A native speaker in a blind A/B test could not tell the fine-tuned NLLB adapter from the zero-shot base model (50% win rate, 22 decided pairs), and rated 30% of all outputs "both bad" regardless of model. The BLEU gains are real and reproducible; they are not perceptible. Details in [`docs/RESULTS_v0.md`](docs/RESULTS_v0.md).

| Track | Status |
|---|---|
| Public Mooré dataset audit (HF + OPUS + Common Voice + Wikipedia) | ✅ done — catalogued in `data/CORPORA.audit.json` |
| Curated parallel corpus **v0.1** | ✅ **205,271 clean pairs / 414,590 direction-rows** — detokenised, LID-gated on every Mooré side, fragment-filtered, pair-level splits, all 4 directions (`data/processed/`, rebuild with `scripts/build_corpus.py`) |
| FLORES-200 eval split | ✅ **1,012 devtest sentences × 4 directions**, fetched from Meta's public mirror (no gating), train/dev decontaminated against it on both sides |
| NLLB LoRA fine-tune | ✅ pipeline validated (shakedown) → full 600M + 3.3B runs on local RTX 4070; adapters in `models/` |
| Automatic evaluation (BLEU / chrF++) | ✅ `scripts/evaluate.py`, before/after in `docs/RESULTS_v0.md` — best gain mos→fra +2.10 BLEU |
| **Human evaluation (blind A/B)** | ✅ **40 native-speaker judgments — and they contradict BLEU.** Fine-tune vs zero-shot is a coin flip (50%, 11/22 decided, p = 0.58); 30% of pairs "both bad"; only mos→eng is a clear win. `eval_app.py`, raw judgments in `data/eval_pack/` |
| ASR corpus (~38k transcribed utterances) | ✅ `scripts/build_asr_corpus.py` → `data/audio/` (see `data/AUDIO_CORPORA.md`) |
| ASR fine-tune (Whisper-small) | ✅ trained — WER 34.1% / CER 11.4% — **superseded, see below** |
| **ASR fine-tune (MMS-1b `mos` adapter)** | ✅ **WER 16.8% / CER 4.3%** on the 564-utterance held-out test — 46% relative cut vs the 31.1% zero-shot MMS baseline. 2.2M trainable params, 3h12m on one RTX 4070 (`scripts/finetune_mms.py`) |
| Published models | ✅ [`Rekin226/nllb-3.3B-moore-lora-v0`](https://huggingface.co/Rekin226/nllb-3.3B-moore-lora-v0) · [`Rekin226/nllb-600M-moore-lora-v0`](https://huggingface.co/Rekin226/nllb-600M-moore-lora-v0) · [`Rekin226/whisper-small-moore-v0`](https://huggingface.co/Rekin226/whisper-small-moore-v0) — ⏳ MMS adapter not yet pushed (`scripts/publish_hf.py`) |
| Demo app (translate + speech→text→translate) | ✅ `app.py` (Gradio, local or HF Space) — now defaults to the fine-tuned MMS adapter when `models/mms-1b-mos-v0/` is present, else stock MMS. Override with `MOORE_ASR_ENGINE=whisper` or `MOORE_MMS_MODEL=...` |
| Common Voice `mos` locale — 750 seed sentences | ✅ **762 French sentences** ready in `data/common_voice_seed/fr_seed_v0.txt` |
| Mooré translations of the seed corpus | ⏳ awaiting native-speaker work → `mos_seed_v0.txt` — **the one step no machine can do** |
| Register-balanced corpus rebuild | ⏳ **now the top priority for translation** — ~89% of the text is Bible-register, the likely cause of the A/B result |

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
| ASR baseline | Meta [MMS-1b-all](https://huggingface.co/facebook/mms-1b-all) with its pretrained `mos` adapter (WER 31.1% zero-shot) |
| ASR fine-tune | Hugging Face `transformers` — adapter-only fine-tune of MMS-1b (`scripts/finetune_mms.py`); Whisper-small also trained for comparison |
| Evaluation | FLORES-200 devtest (BLEU/chrF++) + held-out ASR test split (WER/CER) + native-speaker blind A/B (`eval_app.py`) |
| Compute | One local RTX 4070 (12 GB) — every run in `docs/RESULTS_v0.md` was trained on it |

_Omnilingual ASR, Parakeet-TDT and NeMo were surveyed in Phase 0 (see References) but are not in the v0 pipeline; MMS-1b won the baseline comparison and is what v0 ships._

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

# 4. ASR corpus + fine-tune. MMS is the v0 model (WER 16.8%); Whisper is kept
#    for comparison only (WER 34.1%).
uv run ... python scripts/build_asr_corpus.py
uv run ... python scripts/finetune_mms.py --epochs 3 --output models/mms-1b-mos-v0
uv run ... python scripts/evaluate_asr.py --mms-dir models/mms-1b-mos-v0 \
  --out .logs/asr_mms_ft.json
uv run ... python scripts/finetune_whisper.py --epochs 3 --output models/whisper-small-mos-v0

# 4b. Blind A/B rating app (native-speaker eval of the translation adapters)
uv run --python 3.12 --with streamlit ... streamlit run eval_app.py

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
- [x] Native-speaker blind A/B of fine-tuned vs zero-shot (40 judgments, `eval_app.py`)
- [ ] **Act on the A/B result** — register-balanced corpus rebuild (`--max-per-source`), then retrain and re-run the A/B. Phase 2 is not done until a native speaker can tell the difference.

### Phase 3 — ASR ✅ (v0)
- [x] Assemble 37,654-utterance / 85 h transcribed Mooré audio corpus (`data/AUDIO_CORPORA.md`)
- [x] Fine-tune Whisper-small for Mooré speech→text (WER 34.1% test)
- [x] MMS-1b-all `mos` zero-shot baseline (WER 31.1%)
- [x] **Fine-tune the MMS `mos` adapter on our corpus — WER 16.8% / CER 4.3%, the v0 ASR model**
- [ ] Access to gated CITADEL-BF / goaicorp audio (~2× more data)

### Phase 4 — Release + Upstream
- [x] Gradio demo app (translate + speech→text→translate)
- [x] Publish NLLB adapters + Whisper ASR under `Rekin226/*`
- [ ] Publish the fine-tuned MMS `mos` adapter (`scripts/publish_hf.py` is ready; 8.6 MB artefact)
- [ ] Demo Space — blocked: Gradio Spaces now require an HF PRO subscription (402 on `create_repo`, 2026-08). Runs locally via `python app.py`.
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
