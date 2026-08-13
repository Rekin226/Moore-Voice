# Mooré corpora inventory

Every source that enters the packaged corpus must be listed here with license, size, collection method, and consent status. Non-redistributable sources are referenced by pointer scripts only, not packaged.

## Packaged in `moore_parallel_v0_1.parquet`

Revisions pinned in `moore_voice/corpus.py::REVISIONS`. Counts are unique pairs after cleaning/dedup (each pair is emitted in both directions in the parquet).

| Source | Pairs kept | License | Collection | Redistributable | Notes |
|---|---:|---|---|---|---|
| `michsethowusu/english-moore_sentence-pairs_mt560` | 182,505 | CC-BY-4.0 (per MT560 corpus) | Aligned from MT560; overwhelmingly JW/Bible text | ⚠️ Underlying JW text is "research use only" — **flagged, see below** | En↔Mos. Detokenised. |
| `hfdjobii/mistral-moore-dataset-v2` | 14,847 | none declared | Community (Burkinabè) instruction pairs | Unconfirmed | Fr/En↔Mos extracted from [INST] wrappers; LID-gated. |
| `madoss/nllb-mos-raw` (laser ≥ 1.15) | 7,213 | ODC-BY (NLLB mined data terms) | LASER-mined web bitext | Yes (with attribution) | En↔Mos; LID-gated. |
| OPUS `translatewiki` fr-mos + en-mos | 7,481 | CC-BY-3.0+ (translatewiki.net) | Volunteer UI translations | Yes | Secular register; MediaWiki placeholder rows dropped. |
| FLORES-200 devtest | 2,024 (eval only) | CC-BY-SA-4.0 | Professional translations (Meta) | Yes | Fetched from Meta's public mirror; held out; train/dev decontaminated against it on both sides. |

**mt560/JW licensing caveat:** ~89% of pairs derive from JW religious text whose
original license is research-use-only. The packaged parquet is therefore **not
redistributed as a dataset**; it is built locally by `scripts/build_corpus.py`
from the public upstream repos. Published model *weights* trained on it follow
common practice for research models; a fully-redistributable corpus release
(translatewiki + NLLB-mined + Common Voice seed) is tracked separately.

## Audio

See `AUDIO_CORPORA.md` for the ASR training sources and their status.

## Confirmed (to audit)

| Source | Modality | License | Size (est.) | Redistributable | Notes |
|---|---|---|---|---|---|
| JW300 | Parallel text | JW-specific | ~? | Research use only | Bible/religious; broad domain coverage |
| Lanfrica | Text + audio | Varies | ? | Per-dataset | African language hub |
| AFRIDOC-MT | Parallel text | ? | ~? | Check | Document-level MT corpus |
| FLORES-200 devtest | Parallel text (eval only) | CC-BY-SA-4.0 | 1012 sentences × 200 langs | Yes | Evaluation benchmark |
| Meta Omnilingual ASR Corpus | Audio + transcripts | CC-BY-4.0 (partial) | 3350h across 348 langs | Partial | Check Mooré subset presence |
| Mozilla Common Voice | Audio + transcripts | CC0 | ? for Mooré | Yes | Check current Mooré status |
| Wikipedia mos.wikipedia.org | Monolingual text | CC-BY-SA-3.0 | small | Yes | Baseline monolingual corpus |
| Bible in Mooré (various translations) | Parallel with Fr/En | Per-translation | ~30k verses | Depends | Some public domain, some not |

## Rejected / pointer-only

_(none yet — record here anything found that cannot be redistributed)_

## Audit tasks

- [ ] Run `scripts/audit_existing_corpora.py` — list Hugging Face datasets tagged `mos` or `moore`
- [ ] Check OPUS for any `fr-mos` or `en-mos` bitexts
- [ ] Check Common Voice dashboard — is Mooré an active locale?
- [ ] Check Omnilingual ASR corpus manifest — is `mos_Latn` included?
- [ ] Check AFRIDOC-MT paper for Mooré inclusion

## Provenance schema

Each corpus entry must specify:

- **Source** — where the data came from (URL or organization)
- **License** — SPDX identifier where possible
- **Size** — sentences or hours
- **Redistributable** — Yes / No / Restricted
- **Consent** — for audio: did speakers give informed consent? Was it collected via community organization?
- **Collection method** — scraped / donated / crowdsourced / commissioned
- **Contact** — who to reach for license questions
