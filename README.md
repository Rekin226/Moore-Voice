# Mooré-Voice

**Open translation and speech recognition for Mooré (Mòoré / Mossi, ISO 639-3 `mos`)** — the language of the Mossi people of Burkina Faso, spoken by ~8 million people across Burkina Faso, Côte d'Ivoire, Togo, Ghana, and Mali.

Built on Meta NLLB-200, Meta Omnilingual ASR, and NVIDIA NeMo. Corpus contributions upstream to Common Voice, Lanfrica, and NeMo.

_Repo name on GitHub is `Moore-Voice` (no accent — GitHub restriction). Human name is `Mooré-Voice`._

---

## Status (2026-07)

| Track | Status |
|---|---|
| Zero-shot NLLB-200-3.3B baseline on 6-sentence Mooré test | ✅ done — usable but imperfect (`docs/FINETUNE_PLAN.md`) |
| Public Mooré dataset audit (HF + OPUS + Common Voice + Wikipedia) | ✅ done — 28+ Mooré-specific HF datasets + 2.2M OPUS bitexts catalogued in `data/CORPORA.audit.json` |
| Curated v0 parallel corpus | ✅ **210,455 clean pairs** (mt560 + mistral-v2 + NLLB-filtered) in `data/processed/moore_parallel_v0.parquet` |
| FLORES-200 eval split | ⚠️ needs HF access request to `openlanguagedata/flores_plus` |
| Common Voice `mos` locale — 750 seed sentences | ✅ **762 French sentences** ready in `data/common_voice_seed/fr_seed_v0.txt` |
| Mooré translations of the seed corpus | ⏳ awaiting native-speaker work → `mos_seed_v0.txt` |
| NLLB LoRA fine-tune (v0) | 📋 plan in `docs/FINETUNE_PLAN.md`, awaiting GPU |
| ASR fine-tune (Parakeet / Omnilingual ASR) | 📋 not started |
| Public demo on Hugging Face Space | 📋 not started |

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

**Reproduce the Phase-0 baselines:**

```bash
uv run --python 3.12 \
  --with 'transformers>=4.44' --with 'torch>=2.3,<3' \
  --with 'sentencepiece' --with 'protobuf' \
  python scripts/verify_nllb_baseline.py    # NLLB-200-600M distilled
```

Swap `verify_nllb_baseline.py` for `verify_nllb_3b3.py` to run the 3.3B variant (needs ~6 GB download + ~15 min CPU inference).

**Rebuild the parallel corpus:**

```bash
uv run --python 3.12 \
  --with 'datasets>=2.20' --with 'pandas>=2.0' --with 'pyarrow>=15' \
  --with 'huggingface_hub' \
  python scripts/build_corpus.py
```

Output → `data/processed/moore_parallel_v0.parquet` + `manifest.json`. The eval split needs FLORES access (see the script for the URL).

## How to contribute

- **Translate seed sentences** for Common Voice `mos` locale — see `data/common_voice_seed/README.md`.
- **Report a bad Mooré row** in the curated corpus — open an issue with the row index or the exact source/target text; we tighten the LID heuristic in `build_corpus.py`.
- **Add a source** — put it in `data/CORPORA.md` with license, size, redistributability, and either a loader in `build_corpus.py` or a pointer script if non-redistributable.

## Roadmap

### Phase 0 — Baseline verification ✅
- [x] Run NLLB-200 zero-shot on native-speaker verified sentences
- [x] Audit existing Mooré datasets on HF, OPUS, Common Voice, Wikipedia
- [ ] Test Omnilingual ASR on native Mooré audio

### Phase 1 — Corpus ✅
- [x] Assemble redistributable parallel Fr/En↔Mooré text (**210k pairs**)
- [x] Draft 750-sentence seed corpus for Common Voice `mos` (**762 sentences done**)
- [ ] Translate seed corpus into Mooré and submit to Common Voice
- [ ] Add FLORES-200 devtest as held-out eval (blocked on access request)

### Phase 2 — Fine-tune
- [ ] LoRA fine-tune of NLLB-200-3.3B on the curated corpus (Colab / Azure GPU)
- [ ] Publish HF checkpoint under `Rekin226/nllb-200-3.3B-moore-lora-v0`
- [ ] Live translation demo on Hugging Face Space

### Phase 3 — ASR + Upstream
- [ ] Assemble audio corpus from hfdjobii / CITADEL-BF / goaicorp / Common Voice
- [ ] Fine-tune Parakeet-TDT and Omnilingual ASR
- [ ] Contribute Mooré recipe to NVIDIA/NeMo examples
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
