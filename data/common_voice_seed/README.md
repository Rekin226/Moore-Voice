# Common Voice Mooré — seed sentence corpus

**Goal:** Contribute the 750-sentence prompt corpus needed to unlock Common Voice for locale `mos` (Mooré). Currently Mozilla lists Mooré as registered but not contributable (0 hours, 0 speakers, 0/750 sentences).

**Status:** Pilot batch of 100 French sentences in `fr_seed_pilot_v0.txt`. Awaiting native-speaker style review before generating the full 750.

## Sentence rules (Mozilla Common Voice policy)

- 4-14 words per sentence
- Natural spoken language — things people would actually say aloud
- No PII (personal names, addresses, phone numbers, emails)
- No politics, no religious controversy, no medical/legal claims
- Numbers spelled out (not written as digits)
- No proper nouns except very common place names
- Public domain / CC-0 (original writing satisfies this)

## Pilot corpus (v0) — topic distribution

| Topic                       | Count |
|-----------------------------|------:|
| Greetings & politeness      | 15    |
| Family & community          | 10    |
| Home & meals                | 10    |
| Market & commerce           | 10    |
| Health & clinic             | 10    |
| Weather & seasons           | 8     |
| Agriculture & livestock     | 12    |
| Water & hygiene             | 5     |
| Travel & directions         | 10    |
| Time & numbers              | 10    |
| **Total**                   | **100** |

## Workflow

1. **Review the pilot** — native speaker (you) reviews `fr_seed_pilot_v0.txt`.
   - Any sentences that don't fit the register or that Mooré speakers wouldn't actually say → mark for removal.
   - Any missing everyday scenarios → note them.
2. **Translate the accepted pilot** — you translate the accepted French sentences into Mooré in `mos_seed_pilot_v0.txt`.
3. **Iterate** — I generate the remaining 650 sentences with adjustments based on your review, targeting the same topic balance.
4. **Contribute to Common Voice** — via the Mozilla platform. There's a sentence-collection tool at https://commonvoice.mozilla.org/sentence-collector where sentences get community-validated before entering the voice-recording pool.

## Licensing

All French sentences in this directory are original writing by Claude (Anthropic) on behalf of the Mooré-Voice project and released under **CC0** to satisfy Common Voice licensing requirements. Mooré translations will be licensed the same way when contributed.
