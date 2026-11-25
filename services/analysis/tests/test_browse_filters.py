import pytest
from fastapi import HTTPException

from echora_analysis.browse_filters import predicates


def test_values_are_bound_not_interpolated():
    clauses, values = predicates(
        genre="'; DROP TABLE tracks; --", language="ja,en", translation="en", year_from=2000
    )
    assert "DROP TABLE" not in " ".join(clauses)
    assert values == ["'; DROP TABLE tracks; --", 2000, ["ja", "en"], "en"]
    assert " ".join(clauses).count("%s") == len(values)


@pytest.mark.parametrize("vocals", ["instrumental", "vocal", "female", "male", "unknown"])
def test_vocal_filters_use_current_aggregate(vocals):
    clauses, values = predicates(vocals=vocals)
    assert "current_embeddings" in clauses[0]
    assert "window_index IS NULL" in clauses[0]
    assert not values


def test_unknown_language_and_missing_lyrics():
    clauses, values = predicates(language="unknown,ja", lyrics="missing")
    assert " OR " in clauses[0]
    assert "NOT EXISTS" in clauses[1]
    assert values == [["ja"]]


@pytest.mark.parametrize(
    "options", [{"vocals": "bad"}, {"lyrics": "bad"}, {"year_from": 2020, "year_to": 1990}]
)
def test_invalid_filters(options):
    with pytest.raises(HTTPException):
        predicates(**options)


def test_multiple_values_are_or_within_each_filter_group():
    clauses, values = predicates(
        vocals=["female", "male"],
        lyrics=["available", "ai"],
        genre=["Jazz, Blues", "Rock"],
        translation=["en", "ja"],
        language=["en", "unknown"],
    )
    assert len(clauses) == 5
    assert "&& %s::text[]" in clauses[0]
    assert " OR " in clauses[1]
    assert " OR " in clauses[2]
    assert "target_language=ANY(%s)" in clauses[3]
    assert " OR " in clauses[4]
    assert values == [["Jazz, Blues", "Rock"], ["en"], ["en", "ja"]]
    assert " ".join(clauses).count("%s") == len(values)


def test_repeated_and_legacy_language_values_are_combined_without_duplicates():
    clauses, values = predicates(language=["en,ja", "en", "unknown"], vocals=["vocal", "vocal"])
    assert values == [["en", "ja"]]
    assert " OR " in clauses[0]
    assert clauses[1].count("current_embeddings") == 1


def test_all_statuses_can_be_selected_together():
    clauses, _ = predicates(
        lyrics=["available", "missing"], vocals=["instrumental", "vocal", "unknown"]
    )
    assert " OR " in clauses[0]
    assert "NOT EXISTS" in clauses[0]
    assert clauses[1].count(" OR ") == 2


@pytest.mark.parametrize(
    "options", [{"vocals": ["vocal", "invalid"]}, {"lyrics": ["ai", "invalid"]}]
)
def test_invalid_values_inside_multiselect_are_rejected(options):
    with pytest.raises(HTTPException):
        predicates(**options)


def test_empty_multiselect_adds_no_predicates():
    assert predicates(vocals=[], language=[], lyrics=[], genre=[], translation=[]) == ([], [])


def test_multiselect_values_remain_bound():
    malicious = "x'); DROP TABLE tracks; --"
    clauses, values = predicates(genre=["Jazz", malicious], translation=["en", malicious])
    assert malicious not in " ".join(clauses)
    assert values == [["Jazz", malicious], ["en", malicious]]
