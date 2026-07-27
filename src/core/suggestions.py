"""Prefix word suggestions for the on-screen keyboard.

Dwell typing is slow, so completing a word from its first letters is one of
the highest-leverage accelerators available to a gaze keyboard. The engine is
framework-agnostic and self-contained.

The search employs a two-stage architecture:
1. A fast literal prefix scan using binary search (bisect) and a priority queue
   (heapq) to efficiently retrieve frequency-ranked candidates.
2. A fuzzy correction fallback (Symmetric Delete algorithm + Damerau-Levenshtein) to
   recover candidates from single-edit typos without scanning the entire vocabulary.
"""

from __future__ import annotations

import bisect
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

# A language code maps to one frequency list on disk
_CORPUS_FILES: Final[dict[str, str]] = {
    "EN": "en_50k.txt",
    "PL": "pl_50k.txt",
}

# Fuzzy correction only kicks in once the typed prefix carries enough signal
# to make a one-edit correction meaningful.
_FUZZY_MIN_PREFIX: Final[int] = 4


class Suggestion(NamedTuple):
    """A single completion candidate."""

    word: str
    fuzzy: bool


@dataclass
class CorpusData:
    """Container for the lazily loaded language corpus and its lookup indices."""

    words: list[str]
    rank: dict[str, int]
    delete_index: dict[str, list[str]]


def _deletion_variants(word: str) -> list[str]:
    """Return `word` itself plus every string obtained by deleting one character from it."""

    variants = [word]
    for i in range(len(word)):
        variants.append(word[:i] + word[i + 1:])
    return variants


def _build_delete_index(words: list[str]) -> dict[str, list[str]]:
    """Map every deletion variant to the list of word references that produced it."""

    index: dict[str, list[str]] = {}
    for word in words:
        for variant in _deletion_variants(word):
            index.setdefault(variant, []).append(word)
    return index


def damerau_levenshtein_at_most_1(a: str, b: str) -> bool:
    """Return whether `a` and `b` are within Damerau-Levenshtein distance 1.

    Covers the four single-edit typo shapes:
    substitution, insertion, deletion, and transposition.
    """

    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            if j == i + 1 and a[i] == b[j] and a[j] == b[i]:
                return True
        return False

    # Lengths differ by exactly one
    shorter, longer = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(shorter) and shorter[i] == longer[i]:
        i += 1
    return shorter[i:] == longer[i + 1:]


class SuggestionEngine:
    """Prefix-completes words from frequency corpora and a custom dictionary."""

    def __init__(self, dict_dir: Path, custom_path: Path) -> None:
        self._dict_dir = Path(dict_dir)
        self._custom_path = Path(custom_path)

        self._corpora: dict[str, CorpusData] = {}
        
        self._custom: list[str] = []
        self._custom_sorted: list[str] = []
        self._custom_rank: dict[str, int] = {}
        self._custom_case_map: dict[str, str] = {}
        self._custom_delete_index: dict[str, set[str]] = {}
        self._custom_next_rank: int = -1

        self._load_custom()

    def _corpus(self, lang: str) -> CorpusData:
        """Return cached corpus data for a language, building it lazily on first request."""

        code = lang.upper()
        cached = self._corpora.get(code)
        if cached is not None:
            return cached

        words: list[str] = []
        rank: dict[str, int] = {}
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

        delete_index = _build_delete_index(words)
        loaded = CorpusData(words=words, rank=rank, delete_index=delete_index)

        self._corpora[code] = loaded
        return loaded

    def _add_to_custom_memory(self, word: str, append_chrono: bool = False) -> None:
        """Incrementally update custom indices without rebuilding the entire structure."""

        lowered = word.lower()

        if lowered in self._custom_case_map:
            self._custom.remove(self._custom_case_map[lowered])
        else:
            bisect.insort(self._custom_sorted, lowered)
            for variant in _deletion_variants(lowered):
                self._custom_delete_index.setdefault(variant, set()).add(lowered)

        self._custom_case_map[lowered] = word

        if append_chrono:
            self._custom.append(word)
            self._custom_rank[lowered] = len(self._custom)
        else:
            self._custom.insert(0, word)
            self._custom_rank[lowered] = self._custom_next_rank
            self._custom_next_rank -= 1

    def _load_custom(self) -> None:
        """Load the custom dictionary."""

        try:
            data = json.loads(self._custom_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for w in data:
                    if isinstance(w, str) and w.strip():
                        self._add_to_custom_memory(w.strip(), append_chrono=True)
        except (OSError, ValueError):
            pass

    def add_custom(self, text: str) -> bool:
        """Save a word or phrase to the custom dictionary, newest first, and persist it."""

        word = text.strip()
        if not word:
            return False

        self._add_to_custom_memory(word, append_chrono=False)

        self._custom_path.parent.mkdir(parents=True, exist_ok=True)
        self._custom_path.write_text(json.dumps(self._custom, ensure_ascii=False), encoding="utf-8")
        return True

    def _fuzzy_matches(self, needle: str, lang: str, seen: set[str]) -> list[str]:
        """Recover completions for `needle` that a plain prefix scan would miss."""

        variants = _deletion_variants(needle)

        custom_hits: list[tuple[str, int]] = []
        custom_seen_words: set[str] = set()

        for variant in variants:
            for low_word in self._custom_delete_index.get(variant, ()):
                if low_word in custom_seen_words:
                    continue
                custom_seen_words.add(low_word)
                if low_word == needle or low_word in seen:
                    continue
                if damerau_levenshtein_at_most_1(needle, low_word):
                    custom_hits.append((low_word, self._custom_rank[low_word]))
                    seen.add(low_word)

        custom_hits.sort(key=lambda pair: pair[1])
        final_custom = [self._custom_case_map[w] for w, _ in custom_hits]

        corpus = self._corpus(lang)
        corpus_hits: list[tuple[str, int]] = []
        corpus_seen_words: set[str] = set()

        for variant in variants:
            for word in corpus.delete_index.get(variant, ()):
                if word in corpus_seen_words:
                    continue
                corpus_seen_words.add(word)
                if word == needle or word in seen:
                    continue
                if damerau_levenshtein_at_most_1(needle, word):
                    corpus_hits.append((word, corpus.rank[word]))
                    seen.add(word)

        corpus_hits.sort(key=lambda pair: pair[1])
        final_corpus = [w for w, _ in corpus_hits]

        return final_custom + final_corpus

    def suggest(self, prefix: str, lang: str, k: int = 3) -> list[Suggestion]:
        """Return up to `k` completions of `prefix` for a language."""

        needle = prefix.strip().lower()
        results: list[Suggestion] = []
        seen: set[str] = set()

        # Custom dictionary literal prefix match
        lo = bisect.bisect_left(self._custom_sorted, needle)
        custom_span: list[str] = []

        for low_word in self._custom_sorted[lo:]:
            if not low_word.startswith(needle):
                break
            if low_word != needle and low_word not in seen:
                custom_span.append(low_word)

        if custom_span:
            for low_word in heapq.nsmallest(k, custom_span, key=lambda w: self._custom_rank[w]):
                results.append(Suggestion(self._custom_case_map[low_word], False))
                seen.add(low_word)
                if len(results) >= k:
                    return results

        # Corpus literal prefix match
        corpus = self._corpus(lang)
        lo = bisect.bisect_left(corpus.words, needle)
        span: list[str] = []

        for word in corpus.words[lo:]:
            if not word.startswith(needle):
                break
            if word != needle and word not in seen:
                span.append(word)

        remaining = k - len(results)
        if remaining > 0 and span:
            for word in heapq.nsmallest(remaining, span, key=lambda w: corpus.rank[w]):
                results.append(Suggestion(word, False))
                seen.add(word)

        # Fuzzy correction fallback
        remaining = k - len(results)
        if remaining > 0 and len(needle) >= _FUZZY_MIN_PREFIX:
            for word in self._fuzzy_matches(needle, lang, seen)[:remaining]:
                results.append(Suggestion(word, True))

        return results