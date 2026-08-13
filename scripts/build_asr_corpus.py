"""Assemble the Mooré ASR corpus: 16 kHz mono WAVs + manifest.jsonl.

Sources (see data/AUDIO_CORPORA.md for license/consent status):
  - hfdjobii/tts-moore-femme   — 10,680 single-speaker read utterances w/ transcripts
  - hfdjobii/tts-moore-homme   — 938 male utterances (transcripts spot-checked;
                                  rows whose transcript fails the Mooré LID gate
                                  are dropped — some are French translations)
  - Minervus00/moore_audio_data — ~27k utterances w/ transcripts (parquet, wav bytes)

Output:
  data/audio/wavs/<source>/<id>.wav      (16 kHz mono PCM16)
  data/audio/manifest.jsonl              one row per utterance:
      {"id", "path", "text", "duration_s", "source", "split"}

Filters: duration 0.5–30 s, transcript passes pair sanity (non-empty, has
letters), Mooré LID on transcript (femme corpus is trusted; others gated),
dedup on (text, duration rounded to 0.1 s).

Split: 97/1.5/1.5 train/dev/test, seeded, grouped by transcript so identical
sentences never straddle splits.

Run:
    uv run --python 3.12 --with 'datasets>=2.20' --with huggingface_hub \
      --with pandas --with pyarrow --with soundfile --with librosa \
      python scripts/build_asr_corpus.py
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moore_voice.text import looks_moore, norm_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "data" / "audio"
WAV_DIR = AUDIO_DIR / "wavs"
SEED = 42
MIN_DUR, MAX_DUR = 0.5, 30.0
TARGET_SR = 16_000


def resample_to_16k_mono(data, sr):
    import numpy as np

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        data = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
    peak = max(abs(data.max()), abs(data.min()), 1e-9)
    if peak > 1.0:
        data = data / peak
    return (data * 32767).astype(np.int16)


def write_wav(path: Path, pcm16, sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def clean_transcript(s: str) -> str:
    s = norm_text(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_hfdjobii(repo: str, source_tag: str, trust_lid: bool) -> list[dict]:
    """Snapshot-download a hfdjobii TTS set and convert per metadata.csv."""
    import time

    import pandas as pd
    import soundfile as sf
    from huggingface_hub import snapshot_download

    print(f"[fetch] {repo}")
    # Low concurrency + retry: these repos are thousands of small files and
    # the Hub rate-limits aggressive snapshot pulls (HTTP 429).
    root = None
    for attempt in range(8):
        try:
            root = Path(snapshot_download(repo, repo_type="dataset", max_workers=2))
            break
        except Exception as e:
            wait = min(300, 30 * (attempt + 1))
            print(f"       retry {attempt + 1}/8 in {wait}s ({type(e).__name__}: {str(e)[:80]})")
            time.sleep(wait)
    if root is None:
        print(f"       ! giving up on {repo}")
        return []
    meta = pd.read_csv(root / "metadata.csv")
    rows = []
    skipped_lid = 0
    for i, r in meta.iterrows():
        text = clean_transcript(str(r["sentence"]))
        if not text or not any(c.isalpha() for c in text):
            continue
        if not trust_lid and not looks_moore(text):
            skipped_lid += 1
            continue
        src = root / r["file_name"]
        if not src.exists():
            continue
        try:
            data, sr = sf.read(src)
        except Exception:
            continue
        dur = len(data) / sr
        if not (MIN_DUR <= dur <= MAX_DUR):
            continue
        out = WAV_DIR / source_tag / f"{source_tag}_{i:06d}.wav"
        if not out.exists():
            write_wav(out, resample_to_16k_mono(data, sr))
        rows.append({"id": f"{source_tag}_{i:06d}",
                     "path": str(out.relative_to(AUDIO_DIR)),
                     "text": text, "duration_s": round(dur, 2),
                     "source": source_tag})
    print(f"       kept {len(rows):,} / {len(meta):,} (lid-dropped {skipped_lid})")
    return rows


def load_minervus() -> list[dict]:
    """Minervus00/moore_audio_data — parquet shards with wav bytes + text."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import snapshot_download

    repo = "Minervus00/moore_audio_data"
    print(f"[fetch] {repo} (~13 GB)")
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["data/*"]))
    rows = []
    skipped_lid = 0
    n = 0
    for shard in sorted((root / "data").glob("*.parquet")):
        t = pq.read_table(shard)
        for audio, text in zip(t["bytes"], t["text"], strict=False):
            n += 1
            txt = clean_transcript(str(text))
            if not txt or not any(c.isalpha() for c in txt):
                continue
            if not looks_moore(txt):
                skipped_lid += 1
                continue
            try:
                data, sr = sf.read(io.BytesIO(audio.as_py()))
            except Exception:
                continue
            dur = len(data) / sr
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
            out = WAV_DIR / "minervus" / f"minervus_{n:06d}.wav"
            if not out.exists():
                write_wav(out, resample_to_16k_mono(data, sr))
            rows.append({"id": f"minervus_{n:06d}",
                         "path": str(out.relative_to(AUDIO_DIR)),
                         "text": txt, "duration_s": round(dur, 2),
                         "source": "minervus"})
        print(f"       {shard.name}: total kept so far {len(rows):,}")
    print(f"       kept {len(rows):,} / {n:,} (lid-dropped {skipped_lid})")
    return rows


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rows += load_hfdjobii("hfdjobii/tts-moore-femme", "femme", trust_lid=True)
    rows += load_hfdjobii("hfdjobii/tts-moore-homme", "homme", trust_lid=False)
    rows += load_minervus()

    # Dedup identical (text, duration) — clusters datasets re-shard femme.
    seen: set = set()
    uniq = []
    for r in rows:
        key = (r["text"].casefold(), round(r["duration_s"], 1))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    print(f"[dedup] {len(rows):,} → {len(uniq):,}")

    # Split grouped by transcript.
    rng = random.Random(SEED)
    by_text: dict[str, list[dict]] = {}
    for r in uniq:
        by_text.setdefault(r["text"].casefold(), []).append(r)
    groups = list(by_text.values())
    rng.shuffle(groups)
    n_eval = max(100, int(len(uniq) * 0.015))
    dev: list[dict] = []
    test: list[dict] = []
    train: list[dict] = []
    for g in groups:
        if len(dev) < n_eval:
            dev.extend(g)
        elif len(test) < n_eval:
            test.extend(g)
        else:
            train.extend(g)
    for split, part in [("train", train), ("dev", dev), ("test", test)]:
        for r in part:
            r["split"] = split

    all_rows = train + dev + test
    with open(AUDIO_DIR / "manifest.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(r["duration_s"] for r in all_rows) / 3600
    manifest_meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "utterances": len(all_rows),
        "hours": round(hours, 2),
        "per_split": dict(Counter(r["split"] for r in all_rows)),
        "per_source": dict(Counter(r["source"] for r in all_rows)),
        "sr": TARGET_SR,
        "duration_filter_s": [MIN_DUR, MAX_DUR],
    }
    (AUDIO_DIR / "manifest_meta.json").write_text(json.dumps(manifest_meta, indent=2))
    print(json.dumps(manifest_meta, indent=2))


if __name__ == "__main__":
    main()
