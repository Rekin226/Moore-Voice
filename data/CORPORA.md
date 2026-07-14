# Mooré corpora inventory

Every source that enters the packaged corpus must be listed here with license, size, collection method, and consent status. Non-redistributable sources are referenced by pointer scripts only, not packaged.

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
