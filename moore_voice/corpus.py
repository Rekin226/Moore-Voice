"""Mooré-Voice parallel corpus assembly (v0.1).

Design:
  - Every loader yields canonical *pairs*: {"xx_lang", "xx_text", "mos_text",
    "source"} where xx is the non-Mooré side (eng_Latn / fra_Latn).
  - All text goes through `clean_text` (norm + detok); every pair through
    `pair_ok` and a Mooré-side `looks_moore` LID gate.
  - Split assignment happens at PAIR level, then each pair is emitted in both
    directions (X→mos and mos→X) so the two directions can never straddle a
    split boundary.
  - FLORES-200 devtest (downloaded from Meta's public NLLB mirror,
    CC-BY-SA-4.0) is the held-out eval split, and its sentences are excluded
    from train/dev on BOTH sides (source or target match ⇒ dropped).

Dataset revisions are pinned for reproducibility.
"""

from __future__ import annotations

import io
import json
import random
import re
import tarfile
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .text import clean_text, looks_moore, pair_ok

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"

REVISIONS = {
    "michsethowusu/english-moore_sentence-pairs_mt560": "5f33a5aaeda1b1dd617eea6168d9802188ab223a",
    "hfdjobii/mistral-moore-dataset-v2": "96c07de37d0c28e86b70764ab89a4c69227e9c41",
    "madoss/nllb-mos-raw": "069e0c02d95d22b977d8169585ca7e70d6ffe7de",
}

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
TW_EN_MOS_URL = "https://object.pouta.csc.fi/OPUS-translatewiki/v2026-07-01/moses/en-mos.txt.zip"
# fr-mos has no moses release; extracted via opustools from the XCES XML.
TW_FR_MOS_RELEASE = "v2025-01-01"

MAX_WORDS = 200
SEED = 42
DEV_FRACTION = 0.05


# ---------- helpers -----------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=300) as r:
        dest.write_bytes(r.read())
    return dest


def _pair(xx_lang: str, xx_text: str, mos_text: str, source: str) -> dict | None:
    xx, mos = clean_text(xx_text), clean_text(mos_text)
    if not pair_ok(xx, mos, MAX_WORDS):
        return None
    if not looks_moore(mos):
        return None
    return {"xx_lang": xx_lang, "xx_text": xx, "mos_text": mos, "source": source}


# ---------- loaders (each returns list of canonical pairs) --------------------


def load_mt560() -> list[dict]:
    from datasets import load_dataset

    name = "michsethowusu/english-moore_sentence-pairs_mt560"
    print(f"[load] {name}")
    ds = load_dataset(name, split="train", revision=REVISIONS[name])
    out = []
    for r in ds:
        p = _pair("eng_Latn", r["eng"], r["mos"], "mt560")
        if p:
            out.append(p)
    print(f"       kept {len(out):,} / {len(ds):,}")
    return out


MISTRAL_TO_MOS_RE = re.compile(
    r"\[INST\].*?"
    r"(?:Give the (?:moore|mooré) translation of"
    r"|[EÉ]cris en (?:moore|mooré)"
    r"|Traduis en (?:moore|mooré)"
    r"|Translate to (?:moore|mooré))"
    r"\s*[:\-]?\s*(?P<src>.+?)\s*\[/INST\]\s*(?P<tgt>.+?)(?:</s>|$)",
    re.DOTALL,
)
MISTRAL_TO_FR_RE = re.compile(
    r"\[INST\].*?"
    r"(?:Traduis en fran[cç]ais|Translate to French)"
    r"\s*[:\-]?\s*(?P<src>.+?)\s*\[/INST\]\s*(?P<tgt>.+?)(?:</s>|$)",
    re.DOTALL,
)

_FR_HINTS = {"le", "la", "les", "un", "une", "des", "je", "tu", "il", "elle",
             "nous", "vous", "ils", "elles", "à", "de", "et", "en", "pour",
             "qui", "que", "avec", "sans", "sur", "sous", "est", "sont", "c'est"}
_EN_HINTS = {"the", "a", "an", "is", "are", "was", "were", "i", "you", "he",
             "she", "we", "they", "of", "in", "to", "for", "with", "and",
             "or", "but", "have", "has", "had", "will", "would", "it", "this"}


def detect_fr_en(text: str) -> str | None:
    tokens = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
    if not tokens:
        return None
    fr = sum(1 for t in tokens if t in _FR_HINTS)
    en = sum(1 for t in tokens if t in _EN_HINTS)
    if fr == en == 0:
        return None
    return "fra_Latn" if fr >= en else "eng_Latn"


def load_mistral_v2() -> list[dict]:
    from datasets import load_dataset

    name = "hfdjobii/mistral-moore-dataset-v2"
    print(f"[load] {name}")
    ds = load_dataset(name, split="train", revision=REVISIONS[name])
    out = []
    for r in ds:
        text = r["text"]
        m = MISTRAL_TO_MOS_RE.search(text)
        if m:
            lang = detect_fr_en(m.group("src"))
            if lang is None:
                continue
            p = _pair(lang, m.group("src"), m.group("tgt"), "mistral-v2")
            if p:
                out.append(p)
            continue
        m = MISTRAL_TO_FR_RE.search(text)
        if m:
            # mos is the instruction source here; French is the answer.
            p = _pair("fra_Latn", m.group("tgt"), m.group("src"), "mistral-v2")
            if p:
                out.append(p)
    print(f"       kept {len(out):,} / {len(ds):,}")
    return out


def load_nllb_mined(threshold: float = 1.15) -> list[dict]:
    from datasets import load_dataset

    name = "madoss/nllb-mos-raw"
    print(f"[load] {name} (laser >= {threshold})")
    ds = load_dataset(name, split="train", revision=REVISIONS[name])
    out = []
    for r in ds:
        if r["laser_score"] < threshold:
            continue
        p = _pair("eng_Latn", r["eng_Latn"], r["mos_Latn"], "nllb-mined")
        if p:
            out.append(p)
    print(f"       kept {len(out):,}")
    return out


def load_translatewiki() -> list[dict]:
    """translatewiki.net UI strings (CC-BY-3.0+, secular register).

    en-mos: OPUS moses zip. fr-mos: pre-extracted text files (opustools) —
    run scripts/fetch_translatewiki_fr.py if missing.
    """
    out: list[dict] = []

    zpath = _download(TW_EN_MOS_URL, RAW_DIR / "tw_en-mos.txt.zip")
    with zipfile.ZipFile(zpath) as z:
        en = z.read("translatewiki.en-mos.en").decode().splitlines()
        mos = z.read("translatewiki.en-mos.mos").decode().splitlines()
    kept = 0
    for e, m in zip(en, mos, strict=False):
        p = _pair("eng_Latn", e, m, "translatewiki")
        if p:
            out.append(p)
            kept += 1
    print(f"[load] translatewiki en-mos kept {kept:,} / {len(en):,}")

    fr_f = RAW_DIR / "tw.fr-mos.fr"
    mos_f = RAW_DIR / "tw.fr-mos.mos"
    if fr_f.exists() and mos_f.exists():
        fr = fr_f.read_text().splitlines()
        mos = mos_f.read_text().splitlines()
        kept = 0
        for f, m in zip(fr, mos, strict=False):
            p = _pair("fra_Latn", f, m, "translatewiki")
            if p:
                out.append(p)
                kept += 1
        print(f"[load] translatewiki fr-mos kept {kept:,} / {len(fr):,}")
    else:
        print("[load] translatewiki fr-mos files missing — run scripts/fetch_translatewiki_fr.py")
    return out


def load_flores_eval() -> list[dict]:
    """FLORES-200 devtest from Meta's public mirror. Held-out eval only."""
    tarball = _download(FLORES_URL, RAW_DIR / "flores200_dataset.tar.gz")
    with tarfile.open(tarball) as tf:
        by_name = {m.name.lstrip("./"): m for m in tf.getmembers()}

        def lines(member: str) -> list[str]:
            f = tf.extractfile(by_name[member])
            assert f is not None, member
            return io.TextIOWrapper(f, encoding="utf-8").read().splitlines()

        eng = lines("flores200_dataset/devtest/eng_Latn.devtest")
        fra = lines("flores200_dataset/devtest/fra_Latn.devtest")
        mos = lines("flores200_dataset/devtest/mos_Latn.devtest")
    assert len(eng) == len(fra) == len(mos) == 1012, "unexpected FLORES size"
    out = []
    for e, f, m in zip(eng, fra, mos, strict=False):
        e, f, m = clean_text(e), clean_text(f), clean_text(m)
        if all([e, f, m]):
            out.append({"xx_lang": "eng_Latn", "xx_text": e, "mos_text": m,
                        "source": "flores200-devtest"})
            out.append({"xx_lang": "fra_Latn", "xx_text": f, "mos_text": m,
                        "source": "flores200-devtest"})
    print(f"[load] FLORES-200 devtest: {len(out):,} eval pairs")
    return out


# ---------- assembly ----------------------------------------------------------


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        key = (p["xx_text"].casefold(), p["mos_text"].casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def remove_eval_overlap(pairs: list[dict], eval_pairs: list[dict]) -> list[dict]:
    """Drop any train pair matching an eval sentence on EITHER side."""
    eval_xx = {p["xx_text"].casefold() for p in eval_pairs}
    eval_mos = {p["mos_text"].casefold() for p in eval_pairs}
    return [p for p in pairs
            if p["xx_text"].casefold() not in eval_xx
            and p["mos_text"].casefold() not in eval_mos]


def split_pairs(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pair-level dev split. Pairs sharing a Mooré sentence land in the same
    split (prevents target-side leakage between train and dev)."""
    rng = random.Random(SEED)
    by_mos: dict[str, list[dict]] = {}
    for p in pairs:
        by_mos.setdefault(p["mos_text"].casefold(), []).append(p)
    groups = list(by_mos.values())
    rng.shuffle(groups)
    n_dev_pairs = max(500, int(len(pairs) * DEV_FRACTION))
    dev: list[dict] = []
    train: list[dict] = []
    for g in groups:
        (dev if len(dev) < n_dev_pairs else train).extend(g)
    return train, dev


def to_rows(pairs: list[dict], split: str) -> list[dict]:
    """Emit each pair in both directions."""
    rows = []
    for p in pairs:
        rows.append({"src_lang": p["xx_lang"], "src_text": p["xx_text"],
                     "tgt_lang": "mos_Latn", "tgt_text": p["mos_text"],
                     "source": p["source"], "split": split})
        rows.append({"src_lang": "mos_Latn", "src_text": p["mos_text"],
                     "tgt_lang": p["xx_lang"], "tgt_text": p["xx_text"],
                     "source": p["source"], "split": split})
    return rows


def build(out_name: str = "moore_parallel_v0_1.parquet") -> Path:
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool: list[dict] = []
    pool.extend(load_mt560())
    pool.extend(load_mistral_v2())
    pool.extend(load_nllb_mined())
    pool.extend(load_translatewiki())

    pre = len(pool)
    pool = dedupe_pairs(pool)
    print(f"[dedup] {pre:,} → {len(pool):,} pairs")

    eval_pairs = load_flores_eval()
    pre = len(pool)
    pool = remove_eval_overlap(pool, eval_pairs)
    print(f"[decontaminate] dropped {pre - len(pool):,} pairs overlapping FLORES")

    train_pairs, dev_pairs = split_pairs(pool)
    rows = (to_rows(train_pairs, "train")
            + to_rows(dev_pairs, "dev")
            + to_rows(eval_pairs, "eval"))

    df = pd.DataFrame(rows)
    out = OUT_DIR / out_name
    df.to_parquet(out, index=False)

    per_source = Counter((r["source"], r["split"]) for r in rows)
    per_dir = Counter((r["src_lang"], r["tgt_lang"], r["split"]) for r in rows)
    manifest = {
        "version": "v0.1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "output": str(out.relative_to(REPO_ROOT)),
        "total_rows": len(df),
        "unique_pairs": {"train": len(train_pairs), "dev": len(dev_pairs),
                         "eval": len(eval_pairs)},
        "per_split": dict(Counter(r["split"] for r in rows)),
        "per_source_split": {f"{s}::{sp}": n for (s, sp), n in sorted(per_source.items())},
        "per_direction_split": {f"{a}→{b}::{sp}": n for (a, b, sp), n in sorted(per_dir.items())},
        "revisions": REVISIONS,
        "filters": {
            "nllb_laser_score_min": 1.15,
            "max_words_per_side": MAX_WORDS,
            "moore_lid": "all mos sides (diacritic or function-word ratio >= 0.15)",
            "detokenized": True,
            "fragment_filter": "tgt<=2w & src>=5w (both orientations)",
            "wiki_noise_filter": True,
            "dedup_key": "(casefold xx, casefold mos) pair",
            "dev_split": "pair-level, grouped by mos sentence, seed 42",
            "flores_decontamination": "either-side sentence match removed from train/dev",
            "directions": "each pair emitted X→mos and mos→X",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest["per_split"], indent=2))
    print(f"[write] {out}")
    return out
