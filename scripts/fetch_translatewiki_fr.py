"""Extract translatewiki fr-mos bitext from OPUS XCES XML into data/raw/.

fr-mos has no moses release on OPUS, so this uses opustools to align the
XML release. en-mos (which does have a moses zip) is fetched directly by
moore_voice.corpus.load_translatewiki.

Run:
    uv run --python 3.12 --with opustools python scripts/fetch_translatewiki_fr.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from opustools import OpusRead

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RELEASE = "v2025-01-01"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fr_out = RAW_DIR / "tw.fr-mos.fr"
    mos_out = RAW_DIR / "tw.fr-mos.mos"
    if fr_out.exists() and mos_out.exists():
        print(f"already present: {fr_out}, {mos_out}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        reader = OpusRead(
            directory="translatewiki",
            source="fr",
            target="mos",
            release=RELEASE,
            preprocess="xml",
            write_mode="moses",
            write=[str(fr_out), str(mos_out)],
            download_dir=tmp,
            suppress_prompts=True,
        )
        reader.printPairs()
    n = len(fr_out.read_text().splitlines())
    print(f"wrote {n:,} fr-mos pairs → {fr_out}")


if __name__ == "__main__":
    main()
