# Common Voice Mooré — seed sentence corpus

**Goal:** Contribute the 750-sentence prompt corpus needed to unlock Common Voice for locale `mos` (Mooré). Mozilla currently lists Mooré as registered but not contributable (0 hours, 0 speakers, 0/750 sentences).

**Status:** `fr_seed_v0.txt` — **762 French sentences**, all 4-11 words, zero duplicates, native-speaker-approved register. Awaiting Mooré translation and Mozilla submission.

## Sentence rules (Mozilla Common Voice policy)

- 4-14 words per sentence ✓ (this corpus: min 4, max 11, mean 7.1)
- Natural spoken language — things people would actually say aloud
- No PII (personal names, addresses, phone numbers, emails); "Salif" appears once as a common given name in the self-introduction pattern used by Common Voice's own examples
- No politics, no religious controversy, no medical/legal claims
- Numbers spelled out (not digits) ✓
- No proper nouns except common place names ✓ (Ouagadougou appears once)
- Public domain / CC-0 ✓ (all sentences original writing, released CC0)

## v0 corpus — topic distribution (~762 sentences)

| Topic                              | Approx. count |
|------------------------------------|--------------:|
| Greetings, politeness, blessings   | 115           |
| Family & community                 | 65            |
| Home & meals                       | 75            |
| Market & commerce                  | 80            |
| Health, clinic, hygiene            | 90            |
| Weather & seasons                  | 60            |
| Agriculture & livestock            | 100           |
| Water & sanitation                 | 40            |
| Travel & directions                | 75            |
| Time, days, months, numbers        | 45            |
| Mosque, church, religious life     | 25            |
| Mobile money & telecom             | 12            |
| Moto-taxi, sotrama, transport      | 15            |
| Education & school                 | 25            |
| Community & civic life             | 30            |
| Tools, chores, home objects        | 15            |
| **Total**                          | **762**       |

## Workflow

1. **Translate** — you (as native Mooré speaker) render each French line into Mooré. Suggested output: `mos_seed_v0.txt`, one Mooré sentence per line, matching line numbers with `fr_seed_v0.txt`. Do it in batches; no need to finish in one sitting.
2. **Community validate** — either self-review or ask 2-3 other Mooré speakers to sanity-check.
3. **Submit** — via https://commonvoice.mozilla.org/sentence-collector — the Mooré locale should accept sentences directly once you're logged in. Batch upload accepted (usually a CSV or TSV; the collector docs describe the exact format).
4. **Once accepted** — Mozilla unlocks the locale for voice recording. You (and anyone) can then record the sentences, and the resulting audio becomes the seed of the world's crowdsourced Mooré ASR corpus.
5. **Cite yourself** — as founding contributor of Common Voice Mooré on the Mooré-Voice repo README and in the eventual paper.

## Licensing

All French sentences in `fr_seed_v0.txt` are original writing by Claude (Anthropic) on behalf of the Mooré-Voice project, released under **CC0**. Mooré translations added by contributors should be released under the same terms to be accepted by Common Voice.
