"""Blind A/B accuracy rating app for Mooré-Voice — Streamlit.

Shows each source sentence with two translations (zero-shot vs fine-tuned,
random order, unlabeled). The rater picks which is more accurate. Ratings are
saved per rater; the summary reveals system identities and win rates only
after all items are judged.

Run:
    uv run --python 3.12 --with streamlit streamlit run eval_app.py

Requires data/eval_pack/eval_pack.jsonl (scripts/generate_eval_pack.py).
Ratings land in data/eval_pack/ratings_<rater>.jsonl — one line per judgment,
safe to stop and resume anytime.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

PACK = Path(__file__).parent / "data" / "eval_pack" / "eval_pack.jsonl"

LANG_NAMES = {"mos_Latn": "Mooré", "fra_Latn": "Français", "eng_Latn": "English"}
CHOICES = ["A is better", "B is better", "Both equally good", "Both bad"]


@st.cache_data
def load_pack() -> list[dict]:
    return [json.loads(line) for line in open(PACK)]


def ratings_path(rater: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", rater.lower()).strip("-") or "anonymous"
    return PACK.parent / f"ratings_{slug}.jsonl"


def load_ratings(rater: str) -> dict[int, dict]:
    p = ratings_path(rater)
    if not p.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, open(p))}


def main() -> None:
    st.set_page_config(page_title="Mooré-Voice accuracy test", page_icon="🗣️")
    st.title("🗣️ Mooré-Voice — blind accuracy test")

    if not PACK.exists():
        st.error("Missing data/eval_pack/eval_pack.jsonl — run "
                 "`python scripts/generate_eval_pack.py` first.")
        return

    items = load_pack()
    rater = st.text_input("Your name (for the results file):", value="")
    if not rater.strip():
        st.info("Enter your name to begin. You will see two translations per "
                "sentence — **A and B are two different systems in random "
                "order**. Choose the more accurate one. You can stop and "
                "come back anytime; progress is saved.")
        return

    done = load_ratings(rater)
    remaining = [it for it in items if it["id"] not in done]
    st.progress(len(done) / len(items),
                text=f"{len(done)} / {len(items)} judged")

    if remaining:
        it = remaining[0]
        src_name = LANG_NAMES[it["src_lang"]]
        tgt_name = LANG_NAMES[it["tgt_lang"]]
        st.subheader(f"{src_name} → {tgt_name}")
        st.markdown(f"**Source ({src_name}):**\n> {it['src_text']}")
        if it.get("reference"):
            with st.expander(f"Reference translation ({tgt_name}) — professional human"):
                st.markdown(f"> {it['reference']}")
        col_a, col_b = st.columns(2)
        col_a.markdown(f"### A\n> {it['A']}")
        col_b.markdown(f"### B\n> {it['B']}")

        choice = st.radio("Which translation is more accurate?", CHOICES,
                          index=None, key=f"choice_{it['id']}")
        comment = st.text_input("Comment (optional):", key=f"comment_{it['id']}")
        if st.button("Save & next", type="primary", disabled=choice is None):
            with open(ratings_path(rater), "a") as f:
                f.write(json.dumps({
                    "id": it["id"], "rater": rater.strip(), "choice": choice,
                    "comment": comment.strip(),
                    "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                }, ensure_ascii=False) + "\n")
            st.rerun()
    else:
        st.success("All items judged — thank you! Results below.")
        wins = {"lora": 0, "base": 0, "tie": 0, "both_bad": 0}
        per_dir: dict[str, dict[str, int]] = {}
        for it in items:
            r = done[it["id"]]
            if r["choice"] == "A is better":
                winner = it["A_is"]
            elif r["choice"] == "B is better":
                winner = "base" if it["A_is"] == "lora" else "lora"
            elif r["choice"] == "Both equally good":
                winner = "tie"
            else:
                winner = "both_bad"
            wins[winner] += 1
            d = f"{LANG_NAMES[it['src_lang']]}→{LANG_NAMES[it['tgt_lang']]}"
            per_dir.setdefault(d, {"lora": 0, "base": 0, "tie": 0, "both_bad": 0})
            per_dir[d][winner] += 1

        decided = wins["lora"] + wins["base"]
        st.metric("Fine-tuned model win rate (of decided pairs)",
                  f"{100 * wins['lora'] / decided:.0f}%" if decided else "—")
        st.write({"Fine-tuned better": wins["lora"], "Zero-shot better": wins["base"],
                  "Equally good": wins["tie"], "Both bad": wins["both_bad"]})
        st.markdown("**By direction** (fine-tuned / zero-shot / tie / both bad):")
        for d, w in per_dir.items():
            st.write(f"- {d}: {w['lora']} / {w['base']} / {w['tie']} / {w['both_bad']}")
        st.caption("Fine-tuned = NLLB-3.3B + Mooré LoRA · Zero-shot = NLLB-3.3B base. "
                   f"Raw judgments: {ratings_path(rater)}")


main()
