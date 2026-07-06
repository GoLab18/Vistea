"""Prefix word suggestions for the on-screen keyboard.

Dwell typing is slow, so completing a word from its first letters is one of
the highest-leverage accelerators available to a gaze keyboard: a single
suggestion tile can stand in for several individual key dwells. The engine is
deliberately Qt-free and self-contained so the interaction logic can be tested
without a display and reused outside the UI.

Two ranking sources are merged. The frequency corpora (hermitdave word lists,
one ``word count`` line each) provide broad coverage ordered by how common a
word is; the user's own dictionary provides personalised terms and phrases and
always outranks the corpus, since a word the user chose to save is a stronger
signal of intent than population frequency. The corpus is loaded lazily per
language on first use - most sessions touch a single layout, so paying the
parse cost for both up front would be wasteful.

The prefix scan uses ``bisect`` over a sorted word list to isolate the
matching range and ``heapq.nsmallest`` to pull the top-k by frequency rank,
which keeps a lookup cheap enough to run once per activation.
"""

from __future__ import annotations

import bisect
import heapq
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Corpus file naming: a language code maps to one frequency list on disk.

_CORPUS_FILES: Dict[str, str] = {
    "EN": "en_50k.txt",
    "PL": "pl_50k.txt",
}


class SuggestionEngine:
    """Prefix-completes words from frequency corpora and a custom dictionary.

    Corpora are read from ``dict_dir`` (one ``<code>_50k.txt`` per language)
    and the user's saved words persist as a JSON list in ``custom_path``, most
    recent first. The default paths are supplied by the UI layer, not baked in
    here, so the engine stays independent of the application's directory
    layout.
    """

    def __init__(self, dict_dir: Path, custom_path: Path):
        self._dict_dir = Path(dict_dir)
        self._custom_path = Path(custom_path)
        self._corpora: Dict[str, Tuple[List[str], Dict[str, int]]] = {}
        self._custom: List[str] = self._load_custom()

    # corpus loading

    def _corpus(self, lang: str) -> Tuple[List[str], Dict[str, int]]:
        """Return (sorted words, word -> frequency rank) for a language code,
        loading and caching it on first request. A missing list yields an
        empty corpus so the language still works off the custom dictionary."""

        code = lang.upper()
        cached = self._corpora.get(code)
        if cached is not None:
            return cached

        words: List[str] = []
        rank: Dict[str, int] = {}
        name = _CORPUS_FILES.get(code)
        if name is not None:
            path = self._dict_dir / name
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines:
                token = line.split(" ", 1)[0].strip().lower()
                if token and token not in rank:
                    rank[token] = len(rank)
                    words.append(token)
            words.sort()

        loaded = (words, rank)
        self._corpora[code] = loaded
        return loaded

    # custom dictionary

    def _load_custom(self) -> List[str]:
        try:
            data = json.loads(self._custom_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [str(w) for w in data if isinstance(w, str) and w.strip()]

    def add_custom(self, text: str) -> bool:
        """Save a word or phrase to the custom dictionary, newest first, and
        persist it. Trims whitespace, deduplicates case-insensitively (moving
        an existing entry back to the front), and returns ``False`` for empty
        input without writing."""

        word = text.strip()
        if not word:
            return False

        lowered = word.lower()
        self._custom = [w for w in self._custom if w.lower() != lowered]
        self._custom.insert(0, word)

        self._custom_path.parent.mkdir(parents=True, exist_ok=True)
        self._custom_path.write_text(
            json.dumps(self._custom, ensure_ascii=False), encoding="utf-8")
        return True

    # querying

    def suggest(self, prefix: str, lang: str, k: int = 3) -> List[str]:
        """Return up to ``k`` completions of ``prefix`` for a language.

        Case-insensitive; empty and single-character prefixes are allowed. The
        custom dictionary is consulted first (newest entries win), then the
        corpus by ascending frequency rank. Duplicates and an exact match of
        the prefix itself are excluded."""

        needle = prefix.strip().lower()
        results: List[str] = []
        seen: set = set()

        # Custom entries preserve their stored order (newest first) so a
        # freshly added word surfaces immediately.
        for word in self._custom:
            low = word.lower()
            if low.startswith(needle) and low != needle and low not in seen:
                seen.add(low)
                results.append(word)
                if len(results) >= k:
                    return results

        words, rank = self._corpus(lang)
        lo = bisect.bisect_left(words, needle)
        span: List[str] = []
        for word in words[lo:]:
            if not word.startswith(needle):
                break
            if word != needle and word not in seen:
                span.append(word)

        remaining = k - len(results)
        if remaining > 0 and span:
            for word in heapq.nsmallest(remaining, span, key=lambda w: rank[w]):
                results.append(word)

        return results
