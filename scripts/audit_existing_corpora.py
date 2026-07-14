"""Enumerate Mooré (`mos` / `mos_Latn`) resources across public data hubs.

Queries:
  1. Hugging Face Hub datasets tagged mos / mossi / moore
  2. OPUS corpus registry for any bitext involving mos
  3. Mozilla Common Voice — is mos an active locale?
  4. Wikipedia mos edition — article count + total content size
  5. FLORES-200 devtest — confirm mos_Latn is included

No auth required. Prints JSON summary to stdout and writes it to
data/CORPORA.audit.json so we can commit it as a snapshot.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MoreVoice-audit/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_json(url: str, timeout: int = 30):
    return json.loads(_get(url, timeout=timeout).decode("utf-8"))


def hf_datasets_mos() -> list[dict]:
    """List Hugging Face datasets that mention Mooré language codes."""
    hits: dict[str, dict] = {}
    queries = [
        "https://huggingface.co/api/datasets?filter=language:mos&limit=200",
        "https://huggingface.co/api/datasets?search=moore&limit=100",
        "https://huggingface.co/api/datasets?search=mossi&limit=100",
        "https://huggingface.co/api/datasets?search=mos_Latn&limit=100",
    ]
    for q in queries:
        try:
            for row in _get_json(q):
                hits[row["id"]] = {
                    "id": row["id"],
                    "downloads": row.get("downloads"),
                    "likes": row.get("likes"),
                    "tags": row.get("tags", []),
                }
        except Exception as e:
            print(f"[hf] {q} → {e}", file=sys.stderr)
    return sorted(hits.values(), key=lambda r: (r.get("downloads") or 0), reverse=True)


def opus_bitexts_mos() -> list[dict]:
    """OPUS parallel corpora involving mos."""
    urls = [
        "https://opus.nlpl.eu/opusapi/?source=mos&preprocessing=xml&version=latest",
        "https://opus.nlpl.eu/opusapi/?target=mos&preprocessing=xml&version=latest",
    ]
    out: list[dict] = []
    seen: set[tuple] = set()
    for u in urls:
        try:
            data = _get_json(u)
        except Exception as e:
            print(f"[opus] {u} → {e}", file=sys.stderr)
            continue
        for c in data.get("corpora", []):
            key = (c.get("corpus"), c.get("source"), c.get("target"), c.get("version"))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "corpus": c.get("corpus"),
                "source": c.get("source"),
                "target": c.get("target"),
                "alignment_pairs": c.get("alignment_pairs"),
                "source_tokens": c.get("source_tokens"),
                "target_tokens": c.get("target_tokens"),
                "version": c.get("version"),
                "url": c.get("url"),
            })
    return sorted(out, key=lambda r: r.get("alignment_pairs") or 0, reverse=True)


def common_voice_mos() -> dict:
    """Is mos an active Common Voice locale?"""
    try:
        stats = _get_json("https://commonvoice.mozilla.org/api/v1/stats/languages")
        for row in stats:
            if row.get("locale") in {"mos", "mos-BF"}:
                return {"present": True, **row}
        return {"present": False, "checked": len(stats), "note": "mos not in Common Voice locale list"}
    except Exception as e:
        return {"present": None, "error": str(e)}


def wikipedia_mos() -> dict:
    """Size of mos.wikipedia.org."""
    try:
        data = _get_json(
            "https://mos.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"
        )
        return data["query"]["statistics"]
    except Exception as e:
        return {"error": str(e)}


def flores_mos() -> dict:
    """Does FLORES-200 include mos_Latn?"""
    # The FLORES-200 language list is documented; also look on HF.
    try:
        info = _get_json("https://huggingface.co/api/datasets/facebook/flores")
        card = _get("https://huggingface.co/datasets/facebook/flores/raw/main/README.md").decode("utf-8", "replace")
        return {
            "hf_id": info.get("id"),
            "downloads": info.get("downloads"),
            "mos_Latn_in_readme": "mos_Latn" in card or "mos_latn" in card.lower() or " mos " in card.lower(),
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    result = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "sources": {
            "huggingface_datasets": hf_datasets_mos(),
            "opus_bitexts": opus_bitexts_mos(),
            "common_voice": common_voice_mos(),
            "wikipedia_mos": wikipedia_mos(),
            "flores_200": flores_mos(),
        },
    }
    out_path = DATA_DIR / "CORPORA.audit.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    def _summarize() -> None:
        s = result["sources"]
        print("\n=== SUMMARY ===")
        print(f"HuggingFace datasets tagged/matching mos: {len(s['huggingface_datasets'])}")
        for r in s["huggingface_datasets"][:15]:
            print(f"  · {r['id']} (dl={r.get('downloads')}, ♥{r.get('likes')})")
        print(f"\nOPUS bitexts involving mos: {len(s['opus_bitexts'])}")
        for r in s["opus_bitexts"][:15]:
            print(
                f"  · {r['corpus']} {r['source']}↔{r['target']}"
                f"  pairs={r.get('alignment_pairs')}"
            )
        cv = s["common_voice"]
        print(f"\nCommon Voice: present={cv.get('present')}")
        if cv.get("present"):
            print(f"  hours={cv.get('validatedHours')} sentences={cv.get('sentencesCount')}")
        print(f"\nWikipedia mos: {s['wikipedia_mos']}")
        print(f"\nFLORES-200: {s['flores_200']}")

    _summarize()
    print(f"\nFull audit written to {out_path}")


if __name__ == "__main__":
    main()
