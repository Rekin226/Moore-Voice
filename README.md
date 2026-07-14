# Mooré-Voice

**Open translation and speech recognition for Mooré (Mòoré / Mossi, ISO 639-3 `mos`)** — the language of the Mossi people of Burkina Faso, spoken by ~8 million people across Burkina Faso, Côte d'Ivoire, Togo, Ghana, and Mali.

Built on Meta NLLB-200, Meta Omnilingual ASR, and NVIDIA NeMo. Corpus contributions upstream to Common Voice, Lanfrica, and NeMo.

---

## Why

The models exist. **NLLB-200 already supports Mooré as `mos_Latn`**, and Meta's **Omnilingual ASR** (Nov 2025) covers 1,600+ languages including African ones. The bottleneck for real-world Mooré language technology is *not* compute or architecture — it is a curated, licensed, community-validated corpus of parallel text and transcribed audio. This project builds that corpus and the fine-tuning recipes on top.

## Scope (v0)

- **Translation:** French ↔ Mooré, English ↔ Mooré. Fine-tune NLLB-200 on curated Mooré parallel data.
- **Transcription (ASR):** Mooré → text. Fine-tune NVIDIA NeMo Parakeet-TDT and Meta Omnilingual ASR against a Mooré audio corpus.
- **Corpus:** Community-collected + Bible/JW300 + AFRIDOC-MT + government / NGO sources. All redistributable licenses only.

Out of scope for v0: text-to-speech (Mooré TTS is a v1 target), other Voltaic languages (Fulfulde, Dagbani, Dyula).

## Stack

| Layer | Tool |
|---|---|
| Translation baseline | Meta [NLLB-200](https://ai.meta.com/blog/nllb-200-high-quality-machine-translation/) (`mos_Latn`) |
| Translation fine-tune | NVIDIA NeMo NMT or Hugging Face `transformers` |
| ASR baseline | Meta [Omnilingual ASR](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/), NVIDIA [Parakeet-TDT](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/) / [Canary-1B-v2](https://arxiv.org/pdf/2509.14128) |
| ASR fine-tune | NVIDIA NeMo ASR + NeMo Speech Data Processor |
| Evaluation | FLORES-200 devtest (Mooré subset), community held-out set |

## Roadmap

### Phase 0 — Baseline verification (this week)
- [ ] Run NLLB-200 zero-shot Fr→Mooré and Mooré→Fr on 50 native-speaker-verified sentences; report BLEU + chrF + human ratings.
- [ ] Test Omnilingual ASR on 5 minutes of native Mooré audio (clean + noisy); report WER.
- [ ] Audit existing Mooré datasets on Hugging Face, OPUS, JW300, Lanfrica, AFRIDOC-MT. Produce `data/CORPORA.md`.

### Phase 1 — Corpus (weeks 2–8)
- [ ] Assemble redistributable parallel Fr↔Mooré text (target: 100k+ aligned segments).
- [ ] Contribute Mooré prompts + audio to Mozilla Common Voice if not already present.
- [ ] Curate a domain-focused subset (agricultural extension / groundwater terminology — leverages hydro-postdoc domain overlap).

### Phase 2 — Fine-tune (weeks 6–12)
- [ ] NLLB Mooré fine-tune, publish HF checkpoint under `Rekin226/nllb-more-*`.
- [ ] Parakeet Mooré fine-tune, publish HF checkpoint under `Rekin226/parakeet-more-*`.
- [ ] Live demo on Hugging Face Space (following HydroPhysicsAI pattern).

### Phase 3 — Upstream (weeks 10+)
- [ ] Contribute Mooré recipe to NVIDIA/NeMo examples (parallel to physicsnemo groundwater PR).
- [ ] Contribute audio + text corpus to Mozilla Common Voice.
- [ ] Publish preprint (target: EMNLP or LREC).

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
