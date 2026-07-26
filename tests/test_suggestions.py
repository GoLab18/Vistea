"""Word prediction engine tests: corpus ranking, custom dictionary, typo
correction (symmetric delete), degradation."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.suggestions import Suggestion, SuggestionEngine, damerau_levenshtein_at_most_1


def _write_corpus(dict_dir: Path, name: str, lines: list[str]) -> None:
    dict_dir.mkdir(parents=True, exist_ok=True)
    (dict_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _engine(tmp_path: Path, en: list[str] | None = None, pl: list[str] | None = None) -> SuggestionEngine:
    dict_dir = tmp_path / "dict"
    if en is not None:
        _write_corpus(dict_dir, "en_50k.txt", en)
    if pl is not None:
        _write_corpus(dict_dir, "pl_50k.txt", pl)
    return SuggestionEngine(dict_dir, tmp_path / "custom.json")


def _words(suggestions: list[Suggestion]) -> list[str]:
    return [s.word for s in suggestions]


def test_suggestions_ordered_by_frequency_rank(tmp_path: Path) -> None:
    eng = _engine(tmp_path, en=["the 100", "then 80", "there 60", "they 40"])
    assert _words(eng.suggest("the", "EN", k=3)) == ["then", "there", "they"]


def test_case_insensitive_prefix_matching(tmp_path: Path) -> None:
    eng = _engine(tmp_path, en=["Hello 100", "help 50"])
    assert _words(eng.suggest("HE", "EN", k=2)) == ["hello", "help"]


def test_exact_prefix_is_excluded_from_results(tmp_path: Path) -> None:
    eng = _engine(tmp_path, en=["cat 100", "cats 80", "catalog 60"])
    assert "cat" not in _words(eng.suggest("cat", "EN", k=5))
    assert _words(eng.suggest("cat", "EN", k=5)) == ["cats", "catalog"]


def test_suggest_returns_suggestion_namedtuples_with_fuzzy_flag(tmp_path: Path) -> None:
    eng = _engine(tmp_path, en=["the 100"])
    result = eng.suggest("th", "EN", k=3)
    assert result == [Suggestion("the", False)]
    assert result[0].word == "the"
    assert result[0].fuzzy is False


def test_custom_dictionary_outranks_corpus(tmp_path: Path) -> None:
    eng = _engine(tmp_path, en=["team 100", "tea 90"])
    eng.add_custom("teabag")
    assert eng.suggest("tea", "EN", k=3)[0].word == "teabag"


def test_custom_persists_and_reloads_from_json(tmp_path: Path) -> None:
    eng = _engine(tmp_path, en=["word 100"])
    eng.add_custom("wonderful")
    reloaded = SuggestionEngine(tmp_path / "dict", tmp_path / "custom.json")
    assert reloaded.suggest("wo", "EN", k=3)[0].word == "wonderful"


def test_add_custom_dedupes_and_moves_to_front(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.add_custom("alpha")
    eng.add_custom("beta")
    eng.add_custom("alpha")
    assert eng._custom == ["alpha", "beta"]


def test_add_custom_rejects_empty(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    assert eng.add_custom("   ") is False
    assert eng._custom == []


def test_missing_corpus_falls_back_to_custom_only(tmp_path: Path) -> None:
    eng = SuggestionEngine(tmp_path / "dict", tmp_path / "custom.json")

    assert eng.suggest("wh", "EN", k=3) == []
    eng.add_custom("whatever")
    assert _words(eng.suggest("wh", "EN", k=3)) == ["whatever"]
    assert _words(eng.suggest("wh", "PL", k=3)) == ["whatever"]


def test_multiword_phrase_suggested_by_prefix(tmp_path: Path) -> None:
    eng = _engine(tmp_path, en=["thanks 100"])
    eng.add_custom("thank you very much")
    assert eng.suggest("thank", "EN", k=3)[0].word == "thank you very much"


def test_dl_accepts_substitution() -> None:
    assert damerau_levenshtein_at_most_1("rpwer", "rower") is True


def test_dl_accepts_insertion_and_deletion() -> None:
    assert damerau_levenshtein_at_most_1("domr", "dom") is True
    assert damerau_levenshtein_at_most_1("dom", "domr") is True


def test_dl_accepts_adjacent_transposition() -> None:
    assert damerau_levenshtein_at_most_1("orwer", "rower") is True


def test_dl_rejects_distance_two() -> None:
    assert damerau_levenshtein_at_most_1("roxxer", "rower") is False


def test_dl_accepts_identical_strings() -> None:
    assert damerau_levenshtein_at_most_1("rower", "rower") is True


def test_substitution_typo_recovers_word(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["rower 100"])
    result = eng.suggest("rpwer", "PL", k=3)
    assert result == [Suggestion("rower", True)]


def test_insertion_typo_recovers_word(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["dom 100"])
    result = eng.suggest("domr", "PL", k=3)
    assert result == [Suggestion("dom", True)]


def test_duplicated_letter_typo_recovers_word(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["rower 100"])
    result = eng.suggest("rowwer", "PL", k=3)
    assert result == [Suggestion("rower", True)]


def test_transposition_typo_recovers_word(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["rower 100"])
    result = eng.suggest("orwer", "PL", k=3)
    assert result == [Suggestion("rower", True)]


def test_distance_two_typo_is_not_suggested(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["rower 100"])
    assert eng.suggest("roxxer", "PL", k=3) == []


def test_fuzzy_skipped_below_minimum_prefix_length(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["dom 100"])
    assert eng.suggest("dmm", "PL", k=3) == []


def test_fuzzy_only_runs_when_prefix_stage_is_short(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["rower 100", "rowa 200"])
    result = eng.suggest("rowe", "PL", k=1)
    assert result == [Suggestion("rower", False)]


def test_fuzzy_results_are_ranked_by_frequency_after_prefix_results(tmp_path: Path) -> None:
    eng = _engine(tmp_path, pl=["rowa 100", "rowi 50"])
    result = eng.suggest("rowe", "PL", k=3)
    assert all(s.fuzzy for s in result)
    assert _words(result) == ["rowa", "rowi"]


def test_custom_dictionary_included_in_fuzzy_matches(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.add_custom("teabag")
    result = eng.suggest("teebag", "EN", k=3)
    assert result == [Suggestion("teabag", True)]


def test_custom_dictionary_fuzzy_index_updates_after_add(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    assert eng.suggest("teebag", "EN", k=3) == []
    eng.add_custom("teabag")
    assert eng.suggest("teebag", "EN", k=3) == [Suggestion("teabag", True)]


def test_real_pl_corpus_index_builds_and_suggests_within_budget(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    dict_dir = root / "assets" / "dict"
    if not (dict_dir / "pl_50k.txt").exists():
        pytest.skip("real pl_50k.txt corpus not present in this checkout")

    eng = SuggestionEngine(dict_dir, tmp_path / "custom.json")

    start = time.perf_counter()
    eng._corpus("PL")
    build_time = time.perf_counter() - start
    assert build_time < 3.0

    start = time.perf_counter()
    for _ in range(20):
        eng.suggest("rpwer", "PL", k=3)
    avg = (time.perf_counter() - start) / 20.0
    assert avg < 0.020