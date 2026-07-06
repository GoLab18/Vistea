"""Word prediction engine tests: corpus ranking, custom dictionary, degradation."""

from core.suggestions import SuggestionEngine


def _write_corpus(dict_dir, name, lines):
    dict_dir.mkdir(parents=True, exist_ok=True)
    (dict_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _engine(tmp_path, en=None, pl=None):
    dict_dir = tmp_path / "dict"
    if en is not None:
        _write_corpus(dict_dir, "en_50k.txt", en)
    if pl is not None:
        _write_corpus(dict_dir, "pl_50k.txt", pl)
    return SuggestionEngine(dict_dir, tmp_path / "custom.json")


# corpus ranking

def test_suggestions_ordered_by_frequency_rank(tmp_path):
    eng = _engine(tmp_path, en=["the 100", "then 80", "there 60", "they 40"])
    assert eng.suggest("the", "EN", k=3) == ["then", "there", "they"]


def test_case_insensitive_prefix_matching(tmp_path):
    eng = _engine(tmp_path, en=["Hello 100", "help 50"])
    assert eng.suggest("HE", "EN", k=2) == ["hello", "help"]


def test_exact_prefix_is_excluded_from_results(tmp_path):
    eng = _engine(tmp_path, en=["cat 100", "cats 80", "catalog 60"])
    assert "cat" not in eng.suggest("cat", "EN", k=5)
    assert eng.suggest("cat", "EN", k=5) == ["cats", "catalog"]


# custom dictionary priority

def test_custom_dictionary_outranks_corpus(tmp_path):
    eng = _engine(tmp_path, en=["team 100", "tea 90"])
    eng.add_custom("teabag")
    assert eng.suggest("tea", "EN", k=3)[0] == "teabag"


def test_custom_persists_and_reloads_from_json(tmp_path):
    eng = _engine(tmp_path, en=["word 100"])
    eng.add_custom("wonderful")
    reloaded = SuggestionEngine(tmp_path / "dict", tmp_path / "custom.json")
    assert reloaded.suggest("wo", "EN", k=3)[0] == "wonderful"


def test_add_custom_dedupes_and_moves_to_front(tmp_path):
    eng = _engine(tmp_path)
    eng.add_custom("alpha")
    eng.add_custom("beta")
    eng.add_custom("alpha")
    assert eng._custom == ["alpha", "beta"]


def test_add_custom_rejects_empty(tmp_path):
    eng = _engine(tmp_path)
    assert eng.add_custom("   ") is False
    assert eng._custom == []


# graceful degradation

def test_missing_corpus_falls_back_to_custom_only(tmp_path):
    eng = SuggestionEngine(tmp_path / "dict", tmp_path / "custom.json")

    # no corpus files on disk: only the custom dictionary answers
    assert eng.suggest("wh", "EN", k=3) == []
    eng.add_custom("whatever")
    assert eng.suggest("wh", "EN", k=3) == ["whatever"]
    assert eng.suggest("wh", "PL", k=3) == ["whatever"]


# multi-word phrases

def test_multiword_phrase_suggested_by_prefix(tmp_path):
    eng = _engine(tmp_path, en=["thanks 100"])
    eng.add_custom("thank you very much")
    assert eng.suggest("thank", "EN", k=3)[0] == "thank you very much"
