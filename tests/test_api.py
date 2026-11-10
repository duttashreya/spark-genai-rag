from fastapi.testclient import TestClient
from src.api.main import app


def test_api_health():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_api_hybrid_search():
    with TestClient(app) as client:
        response = client.post(
            "/v1/search",
            json={"query": "Spark unified analytics engine", "top_k": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert "latency_ms" in data
        assert data["total_results"] > 0
        assert "content" in data["results"][0]
        assert "rrf_score" in data["results"][0]