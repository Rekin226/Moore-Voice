"""Deep inspection of the two ⚠️ Mooré datasets that need scrutiny.

1. hfdjobii/mistral-moore-dataset-v2 — check if the [INST] system prompt is
   the same across all rows (duplication) or actually diverse fine-tuning data.
2. madoss/nllb-mos-raw — filter by laser_score and sample random rows to
   spot-check whether the LID errors we saw are systemic or edge cases.
"""

from __future__ import annotations

from datasets import load_dataset
from collections import Counter
import random
import re
import sys

random.seed(42)


def inspect_mistral_v2() -> None:
    print("=" * 72)
    print("hfdjobii/mistral-moore-dataset-v2 — deep look")
    print("=" * 72)

    ds = load_dataset("hfdjobii/mistral-moore-dataset-v2", split="train")
    print(f"train n={len(ds)}  cols={ds.column_names}")

    # Length distribution
    lens = [len(row["text"]) for row in ds]
    print(f"length: min={min(lens)}, max={max(lens)}, mean={sum(lens) // len(lens)}")

    # How many rows share the same first 100 chars?  (system-prompt duplication test)
    prefix_counter = Counter(row["text"][:100] for row in ds)
    top5 = prefix_counter.most_common(5)
    print(f"\nunique 100-char prefixes: {len(prefix_counter)}")
    print("top-5 shared prefixes:")
    for pfx, cnt in top5:
        print(f"  ({cnt:>6}×) {pfx!r}")

    # 5 random full samples
    print("\n=== 5 random FULL samples ===")
    for i in random.sample(range(len(ds)), 5):
        text = ds[i]["text"]
        print(f"\n--- row {i} (len={len(text)}) ---")
        print(text[:1500])
        if len(text) > 1500:
            print(f"... [truncated {len(text) - 1500} more chars]")


def inspect_nllb_mos_raw() -> None:
    print("\n" + "=" * 72)
    print("madoss/nllb-mos-raw — laser_score filter + language spot-check")
    print("=" * 72)

    ds = load_dataset("madoss/nllb-mos-raw", split="train")
    print(f"total n={len(ds)}  cols={ds.column_names}")

    scores = [row["laser_score"] for row in ds]
    print(f"laser_score: min={min(scores):.3f}, max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}")

    thresholds = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    print("\nrows above threshold:")
    for t in thresholds:
        n = sum(1 for s in scores if s >= t)
        print(f"  ≥ {t}: {n:>10,}  ({100 * n / len(ds):.1f}%)")

    # Very simple Mooré-vs-not heuristic: presence of Mooré-typical characters
    # (ɛ ɩ ʋ ɔ tildes, or common short function words).
    # This isn't perfect — but flags rows that clearly look nothing like Mooré.
    def looks_moore(s: str) -> bool:
        if not s:
            return False
        moore_chars = set("ɛɩʋɔẽãĩõũʋ̃ɛ̃ɔ̃ɩ̃")
        if any(c in s for c in moore_chars):
            return True
        # Common Mooré function words
        tokens = re.findall(r"\b[A-Za-zɛɩʋɔẽãĩõũ]+\b", s.lower())
        if not tokens:
            return False
        moore_words = {"la", "yaa", "wã", "sẽn", "tɩ", "yãmb", "koom", "y", "m", "a", "n", "n-", "b-"}
        hits = sum(1 for t in tokens if t in moore_words)
        return hits / max(len(tokens), 1) >= 0.15

    # Random 15 samples from the ≥1.15 filtered pool
    filtered = [i for i, s in enumerate(scores) if s >= 1.15]
    sample_idx = random.sample(filtered, min(15, len(filtered)))
    print(f"\n=== 15 random samples from laser_score ≥ 1.15 pool (n={len(filtered):,}) ===")
    moore_looking = 0
    for i in sample_idx:
        row = ds[i]
        mos = row["mos_Latn"]
        looks = looks_moore(mos)
        moore_looking += int(looks)
        flag = "🟢" if looks else "🔴"
        print(f"\n{flag} row {i}  laser={row['laser_score']:.3f}")
        print(f"  en:  {row['eng_Latn'][:200]}")
        print(f"  mos: {mos[:200]}")
    print(f"\nRough Mooré-looking heuristic hit rate: {moore_looking}/{len(sample_idx)}")


def main() -> None:
    try:
        inspect_mistral_v2()
    except Exception as e:
        print(f"mistral inspect failed: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        inspect_nllb_mos_raw()
    except Exception as e:
        print(f"nllb inspect failed: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
