"""Assemble the Mooré-Voice v0 parallel corpus from public HF sources.

Pipeline (per FINETUNE_PLAN.md):
  1. Load each source, cast to schema {src_lang, src_text, tgt_lang, tgt_text, source}.
  2. Mistral-v2: regex-extract the actual instruction, detect direction, keep pair.
  3. NLLB raw: filter laser_score >= 1.15 + Mooré-orthography heuristic.
  4. Normalise whitespace, drop empty, cap at 200 tokens per side.
  5. Deduplicate on (src_text, tgt_text) after casefolding.
  6. Held-out FLORES-200 devtest as eval split.
  7. Random 5% dev, remainder train.
  8. Write parquet + manifest.

Run:
    uv run --python 3.12 --with 'datasets>=2.20' --with 'pandas>=2.0' \
        --with 'pyarrow>=15' python scripts/build_corpus.py

Output:
    data/processed/moore_parallel_v0.parquet
    data/processed/manifest.json
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset

random.seed(42)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- extractors --------------------------------------------------------

MOORE_CHARS = set("ɛɩʋɔẽãĩõũʋ̃ɛ̃ɔ̃ɩ̃")
MOORE_WORDS = {"la", "yaa", "wã", "sẽn", "tɩ", "yãmb", "koom",
               "n-", "b-", "kãnga", "wakat", "biig", "nebã"}


def looks_moore(s: str) -> bool:
    if not s:
        return False
    if any(c in s for c in MOORE_CHARS):
        return True
    tokens = re.findall(r"\b[A-Za-zɛɩʋɔẽãĩõũ]+\b", s.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in MOORE_WORDS)
    return hits / max(len(tokens), 1) >= 0.15


def norm_text(s: str) -> str:
    s = s.replace(" ", " ").replace("​", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- source loaders ----------------------------------------------------

def load_mt560() -> list[dict]:
    print("[load] michsethowusu/english-moore_sentence-pairs_mt560")
    ds = load_dataset("michsethowusu/english-moore_sentence-pairs_mt560", split="train")
    out = []
    for r in ds:
        en, mos = norm_text(r["eng"]), norm_text(r["mos"])
        if not en or not mos:
            continue
        if len(en.split()) > 200 or len(mos.split()) > 200:
            continue
        out.append({
            "src_lang": "eng_Latn", "src_text": en,
            "tgt_lang": "mos_Latn", "tgt_text": mos,
            "source": "mt560",
        })
    print(f"       kept {len(out):,}")
    return out


# Mistral-v2 has an [INST] wrapper. Extract the instruction verb, source, answer.
MISTRAL_INSTR_RE = re.compile(
    r"\[INST\].*?"
    r"(?:Give the (?:moore|mooré) translation of"
    r"|[EÉ]cris en (?:moore|mooré)"
    r"|Traduis en (?:moore|mooré)"
    r"|Translate to (?:moore|mooré))"
    r"\s*[:\-]?\s*(?P<src>.+?)\s*\[/INST\]\s*(?P<tgt>.+?)(?:</s>|$)",
    re.DOTALL,
)
MISTRAL_INSTR_TO_FR_RE = re.compile(
    r"\[INST\].*?"
    r"(?:Traduis en fran[cç]ais|Translate to French)"
    r"\s*[:\-]?\s*(?P<src>.+?)\s*\[/INST\]\s*(?P<tgt>.+?)(?:</s>|$)",
    re.DOTALL,
)


def _detect_lang(text: str) -> str | None:
    """Cheap Fr vs En detector for the Mistral instruction source side."""
    fr_hints = {"le", "la", "les", "un", "une", "des", "je", "tu", "il", "elle",
                "nous", "vous", "ils", "elles", "à", "de", "et", "en", "pour",
                "qui", "que", "avec", "sans", "sur", "sous"}
    en_hints = {"the", "a", "an", "is", "are", "was", "were", "i", "you", "he",
                "she", "we", "they", "of", "in", "to", "for", "with", "and",
                "or", "but", "have", "has", "had", "will", "would"}
    tokens = re.findall(r"[A-Za-zÀ-ÿ']+", text.lower())
    if not tokens:
        return None
    fr = sum(1 for t in tokens if t in fr_hints)
    en = sum(1 for t in tokens if t in en_hints)
    if fr == en == 0:
        # Neither — treat as ambiguous; skip.
        return None
    return "fra_Latn" if fr >= en else "eng_Latn"


def load_mistral_v2() -> list[dict]:
    print("[load] hfdjobii/mistral-moore-dataset-v2")
    ds = load_dataset("hfdjobii/mistral-moore-dataset-v2", split="train")
    out = []
    for r in ds:
        text = r["text"]
        # Try Fr/En → Mooré first.
        m = MISTRAL_INSTR_RE.search(text)
        if m:
            src = norm_text(m.group("src"))
            tgt = norm_text(m.group("tgt"))
            if not src or not tgt or len(src.split()) > 200:
                continue
            src_lang = _detect_lang(src)
            if src_lang is None:
                continue
            out.append({
                "src_lang": src_lang, "src_text": src,
                "tgt_lang": "mos_Latn", "tgt_text": tgt,
                "source": "mistral-v2",
            })
            continue
        # Try Mooré → French (for later reverse direction; keep it).
        m = MISTRAL_INSTR_TO_FR_RE.search(text)
        if m:
            src = norm_text(m.group("src"))
            tgt = norm_text(m.group("tgt"))
            if not src or not tgt:
                continue
            out.append({
                "src_lang": "mos_Latn", "src_text": src,
                "tgt_lang": "fra_Latn", "tgt_text": tgt,
                "source": "mistral-v2-rev",
            })
    print(f"       kept {len(out):,}")
    return out


def load_nllb_filtered(threshold: float = 1.15) -> list[dict]:
    print(f"[load] madoss/nllb-mos-raw (laser_score >= {threshold})")
    ds = load_dataset("madoss/nllb-mos-raw", split="train")
    out = []
    dropped_lid = 0
    for r in ds:
        if r["laser_score"] < threshold:
            continue
        en = norm_text(r["eng_Latn"])
        mos = norm_text(r["mos_Latn"])
        if not en or not mos or len(en.split()) > 200 or len(mos.split()) > 200:
            continue
        if not looks_moore(mos):
            dropped_lid += 1
            continue
        out.append({
            "src_lang": "eng_Latn", "src_text": en,
            "tgt_lang": "mos_Latn", "tgt_text": mos,
            "source": "nllb-mined",
        })
    print(f"       kept {len(out):,}  (dropped {dropped_lid:,} for non-Mooré-looking mos)")
    return out


def load_flores_eval() -> list[dict]:
    """FLORES-200 devtest for held-out evaluation. Both directions."""
    print("[load] openlanguagedata/flores_plus (devtest, mos_Latn ↔ {eng,fra}_Latn)")
    out = []
    try:
        eng = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
        fra = load_dataset("openlanguagedata/flores_plus", "fra_Latn", split="devtest")
        mos = load_dataset("openlanguagedata/flores_plus", "mos_Latn", split="devtest")
    except Exception as e:
        print(f"  ! flores_plus load failed: {e}")
        return []
    n = min(len(eng), len(fra), len(mos))
    for i in range(n):
        e, f, m = norm_text(eng[i]["text"]), norm_text(fra[i]["text"]), norm_text(mos[i]["text"])
        if all([e, f, m]):
            out.append({"src_lang": "eng_Latn", "src_text": e,
                        "tgt_lang": "mos_Latn", "tgt_text": m, "source": "flores-devtest"})
            out.append({"src_lang": "fra_Latn", "src_text": f,
                        "tgt_lang": "mos_Latn", "tgt_text": m, "source": "flores-devtest"})
    print(f"       kept {len(out):,} eval pairs (~{n} sentences × 2 directions)")
    return out


# ---------- assembly ----------------------------------------------------------

def dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = (r["src_text"].casefold(), r["tgt_text"].casefold())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    train_pool: list[dict] = []
    train_pool.extend(load_mt560())
    train_pool.extend(load_mistral_v2())
    train_pool.extend(load_nllb_filtered(1.15))

    pre_dedup = len(train_pool)
    train_pool = dedupe(train_pool)
    print(f"\n[dedup] {pre_dedup:,} → {len(train_pool):,}")

    eval_pool = load_flores_eval()
    # Guard: make sure no FLORES pair leaks into train
    eval_keys = {(r["src_text"].casefold(), r["tgt_text"].casefold()) for r in eval_pool}
    train_pool = [r for r in train_pool if (r["src_text"].casefold(), r["tgt_text"].casefold()) not in eval_keys]

    # Random 5% dev split
    random.shuffle(train_pool)
    n_dev = max(500, len(train_pool) // 20)
    dev = train_pool[:n_dev]
    train = train_pool[n_dev:]

    for split_name, rows in [("train", train), ("dev", dev), ("eval", eval_pool)]:
        for r in rows:
            r["split"] = split_name

    all_rows = train + dev + eval_pool
    df = pd.DataFrame(all_rows)

    out_parquet = OUT_DIR / "moore_parallel_v0.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\n[write] {out_parquet}  rows={len(df):,}  cols={list(df.columns)}")

    # Manifest
    per_source = Counter((r["source"], r["split"]) for r in all_rows)
    per_direction = Counter((r["src_lang"], r["tgt_lang"], r["split"]) for r in all_rows)
    manifest = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "output": str(out_parquet.relative_to(HERE.parent)),
        "total_rows": len(df),
        "per_split": {
            "train": len(train),
            "dev": len(dev),
            "eval": len(eval_pool),
        },
        "per_source_split": {f"{s}::{sp}": n for (s, sp), n in per_source.items()},
        "per_direction_split": {f"{a}→{b}::{sp}": n for (a, b, sp), n in per_direction.items()},
        "filters": {
            "nllb_laser_score_min": 1.15,
            "max_tokens_per_side": 200,
            "dedup_key": "(casefold src, casefold tgt)",
            "flores_leak_check": "eval pairs removed from train pool",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[write] data/processed/manifest.json\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
