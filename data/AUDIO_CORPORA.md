# Mooré audio corpora inventory

Status of every audio source considered for the ASR track. Same policy as
`CORPORA.md`: sources without a clear redistributable license are used for
**model training only** (weights published, audio never repackaged or
redistributed), and are listed here with their access status.

## Used for training (audio not redistributed)

| Source | Utterances | License tag | Access | Notes |
|---|---:|---|---|---|
| `hfdjobii/tts-moore-femme` | 10,680 | none declared | public | Single female speaker, clean read speech, transcripts + SNR/pitch metadata. TTS-grade. |
| `hfdjobii/tts-moore-homme` | 938 | none declared | public | Male speaker; some transcript rows are French — rows failing the Mooré LID gate are dropped. |
| `Minervus00/moore_audio_data` | ~27,000 | none declared | public | 13.2 GB parquet (wav bytes + text). Mirror of CITADEL-BF collection. |

**License caveat:** none of these repos declare a license. They are publicly
posted on Hugging Face with transcripts, uploaded by Burkinabè community
members for exactly this purpose, but formal redistribution rights are
unconfirmed. Consequence: we publish **model weights and manifests only** —
never the audio itself. Before any dataset re-release, contact the uploaders
(tracked as an open issue).

## Gated / pending access

| Source | Size | Status |
|---|---|---|
| `CITADEL-BF-Center/moore_audio_data` | 73 parquet shards | gated — access requested; appears identical to the Minervus00 mirror |
| `goaicorp/moore-speech-bible` | 42 shards, 20.5 GB | gated — access requested; Bible readings, would add ~2× volume |

## Rejected

| Source | Reason |
|---|---|
| `hfdjobii/tts-moore-cluster-{0..4}` | Re-shards of `tts-moore-femme` (same `c0_*` ids) — would duplicate. Dedup guard also catches this. |
| `louisbertson/moore-audio-standardized` | Long-form recordings (up to 250 s) whose "transcripts" are recording titles, not verbatim text. Unusable for supervised ASR as-is. |

## Future

- **Mozilla Common Voice `mos`** — the 762-sentence seed corpus in
  `data/common_voice_seed/` unlocks crowdsourced, CC0 Mooré audio once
  translated and submitted. That will become the preferred (fully licensed)
  training source.
- **Meta Omnilingual ASR corpus** — check `mos` subset presence (CC-BY-4.0
  partial).
