"""Inspect the Mooré audio datasets from the corpus audit.

Uses streaming mode so we don't download tens of GB. Reports for each:
  - splits + total row count (from `info.splits` metadata)
  - column names + column types
  - one sample row (audio metadata only, not the audio bytes)
  - sample rate + duration if audio column is loadable
  - transcript column if present

Feeds into Phase 1.5 planning for ASR fine-tuning.
"""

from __future__ import annotations

from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from huggingface_hub import HfApi
import sys
import traceback

CANDIDATES = [
    "hfdjobii/tts-moore-femme",
    "hfdjobii/tts-moore-homme",
    "hfdjobii/tts-moore-cluster-0",
    "hfdjobii/tts-moore-cluster-1",
    "hfdjobii/tts-moore-cluster-2",
    "hfdjobii/tts-moore-cluster-3",
    "hfdjobii/tts-moore-cluster-4",
    "CITADEL-BF-Center/moore_audio_data",
    "Minervus00/moore_audio_data",
    "louisbertson/moore-audio-standardized",
    "goaicorp/moore-speech-bible",
]


def _info_summary(dataset_id: str) -> dict:
    """Cheap HF Hub metadata lookup — repo size, filenames, siblings."""
    api = HfApi()
    try:
        info = api.dataset_info(dataset_id)
        size_bytes = sum(getattr(s, "size", 0) or 0 for s in info.siblings)
        file_exts: dict[str, int] = {}
        for s in info.siblings:
            ext = s.rfilename.rsplit(".", 1)[-1].lower() if "." in s.rfilename else "(none)"
            file_exts[ext] = file_exts.get(ext, 0) + 1
        return {
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "repo_size_mb": round(size_bytes / (1024 * 1024), 1),
            "file_count": len(info.siblings),
            "top_exts": sorted(file_exts.items(), key=lambda x: -x[1])[:6],
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def inspect(dataset_id: str) -> None:
    print(f"\n{'=' * 72}\n{dataset_id}\n{'=' * 72}")
    meta = _info_summary(dataset_id)
    if "error" in meta:
        print(f"  ! hub metadata error: {meta['error']}")
    else:
        print(f"  hub: dl={meta['downloads']}  ♥{meta['likes']}  "
              f"size={meta['repo_size_mb']} MB  files={meta['file_count']}  "
              f"top_exts={meta['top_exts']}")

    try:
        configs = get_dataset_config_names(dataset_id) or [None]
    except Exception as e:
        print(f"  ! configs error: {type(e).__name__}: {str(e)[:120]}")
        return

    for cfg in configs[:2]:
        cfg_label = cfg or "(default)"
        try:
            splits = get_dataset_split_names(dataset_id, config_name=cfg) if cfg \
                else get_dataset_split_names(dataset_id)
        except Exception as e:
            print(f"  cfg={cfg_label}  split error: {type(e).__name__}: {str(e)[:120]}")
            continue
        print(f"  cfg={cfg_label}  splits={splits}")

        for sp in splits[:1]:
            try:
                ds = load_dataset(dataset_id, cfg, split=sp, streaming=True) if cfg \
                    else load_dataset(dataset_id, split=sp, streaming=True)
                first = next(iter(ds))
                cols = list(first.keys())
                print(f"    split={sp}  cols={cols}")
                for k in cols[:8]:
                    v = first[k]
                    if isinstance(v, dict) and "sampling_rate" in v:
                        arr_len = len(v.get("array") or []) if v.get("array") is not None else None
                        dur = arr_len / v["sampling_rate"] if arr_len else None
                        dur_str = f", ~{dur:.1f}s" if dur else ""
                        print(f"      [{k}] Audio(sr={v['sampling_rate']}"
                              f", path={v.get('path', '?')!s:.60}{dur_str})")
                    elif isinstance(v, bytes):
                        print(f"      [{k}] bytes(len={len(v)})")
                    elif isinstance(v, str):
                        preview = v[:120].replace("\n", " ")
                        print(f"      [{k}] str(len={len(v)}): {preview!r}")
                    else:
                        print(f"      [{k}] {type(v).__name__}: {str(v)[:120]}")
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
