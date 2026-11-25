def test_health(anonymous):
    assert anonymous.get("/api/v1/system/health").json() == {
        "status": "ok",
        "service": "echora-backend",
    }


def test_capabilities(anonymous):
    models = anonymous.get("/api/v1/system/capabilities").json()["models"]
    assert models["muq_mulan"] == {"status": "available", "audio": True, "text": True}
