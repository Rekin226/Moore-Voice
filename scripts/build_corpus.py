"""Build the Mooré-Voice v0.1 parallel corpus.

All logic lives in moore_voice.corpus (importable + unit-tested).

Run:
    uv run --python 3.12 --with 'datasets>=2.20' --with 'pandas>=2.0' \
        --with 'pyarrow>=15' python scripts/build_corpus.py

Optional first (for the fr-mos translatewiki slice):
    uv run --python 3.12 --with opustools python scripts/fetch_translatewiki_fr.py

Output:
    data/processed/moore_parallel_v0_1.parquet
    data/processed/manifest.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moore_voice.corpus import build  # noqa: E402

if __name__ == "__main__":
    build()
