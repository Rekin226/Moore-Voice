"""Load and inspect the top Mooré-specific text datasets on Hugging Face.

Prints for each dataset:
  - configs available
  - split sizes
  - column names
  - 3-5 sample rows (truncated)
  - obvious quality / alignment hints

Focused on TEXT parallel/monolingual datasets useful for translation fine-tuning.
Audio datasets get their own inspector later (bigger download).
"""

from __future__ import annotations

from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
import traceback
import json

# Text-first Mooré datasets from the audit. Skipping audio (bigger download, own inspector).
CANDIDATES = [
    "sawadogosalif/MooreFRCollections",
    "michsethowusu/english-moore_sentence-pairs_mt560",
    "madoss/nllb-mos-raw",
    "hfdjobii/mistral-moore-dataset-v2",
    "michsethowusu/mossi-emotions-corpus",
]


def _trunc(v, n=120):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "…"
    if isinstance(v, (list, dict)):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= n else s[:n] + "…"
    return v


def inspect(dataset_id: str) -> None:
    print(f"\n{'=' * 70}\n{dataset_id}\n{'=' * 70}")
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as e:
        print(f"  ! configs error: {type(e).__name__}: {e}")
        configs = [None]
    for cfg in (configs or [None]):
        cfg_label = cfg or "(default)"
        try:
            splits = get_dataset_split_names(dataset_id, config_name=cfg) if cfg else get_dataset_split_names(dataset_id)
        except Exception as e:
            print(f"  cfg={cfg_label}: split error {type(e).__name__}: {e}")
            splits = []
        print(f"  cfg={cfg_label}  splits={splits}")
        for sp in splits[:2]:
            try:
                ds = load_dataset(dataset_id, cfg, split=sp) if cfg else load_dataset(dataset_id, split=sp)
                cols = ds.column_names
                print(f"    split={sp}  n={len(ds)}  cols={cols}")
                for i, row in enumerate(ds.select(range(min(3, len(ds))))):
                    trunc = {k: _trunc(row[k]) for k in cols[:6]}
                    print(f"      [{i}] {trunc}")
            except Exception as e:
                print(f"    split={sp}: load error {type(e).__name__}: {str(e)[:200]}")


def main() -> None:
    for did in CANDIDATES:
        try:
            inspect(did)
        except Exception:
            print(f"\n{did}: unhandled")
            traceback.print_exc()


if __name__ == "__main__":
    main()
