import backend.search.concepts as concepts
from backend.search.service import shutdown


def test_concept_search_runs_the_lens_for_the_user(client, user, monkeypatch):
    calls = []
    result = {
        "run_id": "r",
        "corpus_size": 1,
        "calibration": "empirical-library-percentile",
        "minimum_percentile": 0.0,
        "representation": "hybrid",
        "scores": {
            "t1": [
                {
                    "name": "Dreamy",
                    "raw_score": 0.5,
                    "percentile": 0.9,
                    "semantic_percentile": 0.8,
                    "lyrics_percentile": None,
                    "lyrics_available": False,
                }
            ]
        },
    }
    monkeypatch.setattr(concepts, "concept_lens", lambda *a: calls.append(a) or result)
    try:
        response = client.post(
            "/api/v1/search/concepts", json={"concepts": ["Dreamy"], "minimum_percentile": 0}
        )
    finally:
        shutdown()
    assert response.status_code == 200, response.text
    assert response.json() == result
    ((user_id, query),) = calls
    assert user_id == user["id"]
    assert query == concepts.LensQuery(["Dreamy"], 0.0, "hybrid")


def test_concept_search_validates_the_body(client):
    assert client.post("/api/v1/search/concepts", json={"concepts": []}).status_code == 422
    assert (
        client.post(
            "/api/v1/search/concepts", json={"concepts": ["a"], "representation": "audio"}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/v1/search/concepts", json={"concepts": ["a"], "other": 1}).status_code
        == 422
    )
