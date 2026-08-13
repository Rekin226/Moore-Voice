"""Text normalisation, detokenisation, and Mooré language-ID heuristics.

Shared by the corpus builder, the fine-tune script, and the demo app so that
train-time and inference-time text processing can never drift apart.
"""

from __future__ import annotations

import re
import unicodedata

# Mooré-specific vowels/diacritics (NFC forms + combining-tilde variants).
MOORE_CHARS = set("ɛɩʋɔẽãĩõũ") | {"ʋ̃", "ɛ̃", "ɔ̃", "ɩ̃"}

# High-frequency Mooré function words / particles. Used only when a sentence
# carries no Mooré-specific diacritic at all (proper Mooré text almost always
# does), so false positives here are second-line noise, not primary signal.
MOORE_WORDS = {
    "la", "yaa", "sẽn", "tɩ", "yãmb", "koom", "wakat", "biig",
    "wã", "kãnga", "nebã", "pʋgẽ", "yell", "naab", "tõnd", "bãmb",
    "ye", "bɩ", "rẽ", "woto", "maan", "paam", "wẽnnaam", "zĩiga",
}

_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TAG_RE = re.compile(r"<[^>]{1,80}>")

# MediaWiki / template noise (translatewiki, some mistral rows).
_WIKI_NOISE_RE = re.compile(r"\$\d|\{\{|\}\}|\[\[|\]\]|__[A-Z]+__|&\w+;")

# Detokenisation: the mt560/translatewiki corpora are moses-tokenised
# ("word , word .", "s ’ est", "tʋm - tʋmdbã"). NLLB expects natural text.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%)\]}»])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{«])\s+")
_APOSTROPHE_RE = re.compile(r"(\w)\s*(['’ʼ])\s*(\w)")
_SPACED_HYPHEN_RE = re.compile(r"(\w) - (\w)")


def norm_text(s: str) -> str:
    """Unicode NFC + collapse whitespace + strip zero-width/nbsp junk."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ").replace("​", "").replace("﻿", "")
    return _WS_RE.sub(" ", s).strip()


def detokenize(s: str) -> str:
    """Undo moses-style tokenisation so training text looks like real text."""
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _SPACE_AFTER_OPEN_RE.sub(r"\1", s)
    s = _APOSTROPHE_RE.sub(r"\1'\3", s)
    s = _SPACED_HYPHEN_RE.sub(r"\1-\2", s)
    return _WS_RE.sub(" ", s).strip()


def clean_text(s: str) -> str:
    """norm + strip URLs/HTML tags + detokenise. The canonical pipeline."""
    s = norm_text(s)
    s = _URL_RE.sub("", s)
    s = _TAG_RE.sub("", s)
    return detokenize(s)


def looks_moore(s: str) -> bool:
    """Heuristic LID: does this string plausibly contain Mooré?"""
    if not s:
        return False
    if any(c in s for c in "ɛɩʋɔẽãĩõũ"):
        return True
    tokens = re.findall(r"[a-zãẽĩõũɛɩʋɔẽ'-]+", s.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in MOORE_WORDS)
    return hits / len(tokens) >= 0.15


def has_wiki_noise(s: str) -> bool:
    """True if the string still carries MediaWiki placeholders/templates."""
    return bool(_WIKI_NOISE_RE.search(s))


def n_words(s: str) -> int:
    return len(s.split())


def n_alpha_tokens(s: str) -> int:
    """Tokens containing at least one letter (any script)."""
    return sum(1 for t in s.split() if any(c.isalpha() for c in t))


def is_fragment_pair(src: str, tgt: str) -> bool:
    """Dictionary-entry artifact: full sentence on one side, 1-2 words on the
    other. Trains the model to truncate — drop."""
    sw, tw = n_words(src), n_words(tgt)
    return (tw <= 2 and sw >= 5) or (sw <= 2 and tw >= 5)


def pair_ok(src: str, tgt: str, max_words: int = 200) -> bool:
    """Structural gate applied to every pair, every source, both sides."""
    if not src or not tgt:
        return False
    if n_words(src) > max_words or n_words(tgt) > max_words:
        return False
    if n_alpha_tokens(src) < 1 or n_alpha_tokens(tgt) < 1:
        return False
    if src.casefold() == tgt.casefold():
        return False
    if is_fragment_pair(src, tgt):
        return False
    if has_wiki_noise(src) or has_wiki_noise(tgt):
        return False
    return True
